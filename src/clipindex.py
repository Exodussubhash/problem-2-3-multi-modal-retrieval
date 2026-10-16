# clipindex.py
import os, sys, glob, json, time, argparse
import numpy as np, cv2, torch, faiss
from transformers import CLIPModel, CLIPProcessor

VID_DIR = os.path.expanduser("~/Downloads/bdd100k_videos")
OUT = "index_out"
MODEL = "openai/clip-vit-large-patch14"
FRAME_TIMES = [2, 6, 10, 14, 22, 30]      # seconds; 10s = BDD100K labeled frame
SHARD_CLIPS = 50
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
DT = torch.float16 if DEV == "mps" else torch.float32

_m = {}
def clip():
    if not _m:
        _m["model"] = CLIPModel.from_pretrained(MODEL, dtype=DT).to(DEV).eval()
        _m["proc"] = CLIPProcessor.from_pretrained(MODEL,use_fast=True)
    return _m["model"], _m["proc"]

@torch.no_grad()
def embed_images(frames, bs=32):
    model, proc = clip()
    out = []
    for i in range(0, len(frames), bs):
        px = proc(images=frames[i:i+bs], return_tensors="pt")["pixel_values"].to(DEV, DT)
        f = model.get_image_features(pixel_values=px).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.vstack(out)

@torch.no_grad()
def embed_text(texts):
    model, proc = clip()
    t = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEV)
    f = model.get_text_features(**t).float()
    return np.ascontiguousarray(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())

def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, ts = [], []
    for t in FRAME_TIMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(t * fps), max(total - 1, 0)))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); ts.append(t)
    cap.release()
    return frames, ts

def stem(p): return os.path.splitext(os.path.basename(p))[0]

def done_clips():
    s = set()
    for p in glob.glob(f"{OUT}/shards/*.jsonl"):
        for line in open(p): s.add(json.loads(line)["clip"])
    return s

def build(n):
    os.makedirs(f"{OUT}/shards", exist_ok=True)
    paths = sorted(glob.glob(f"{VID_DIR}/*.mov") + glob.glob(f"{VID_DIR}/*.mp4"))
    if os.path.exists("labels_slim.json"):      # only clips that have labels
        keep = {os.path.splitext(x["name"])[0] for x in json.load(open("labels_slim.json"))}
        paths = [p for p in paths if stem(p) in keep]
    done = done_clips()
    todo = [p for p in paths[:n] if stem(p) not in done]
    print(f"{len(paths[:n])} clips selected, {len(todo)} to do, device={DEV}")
    shard_id, t0 = len(glob.glob(f"{OUT}/shards/*.npy")), time.time()
    for g in range(0, len(todo), SHARD_CLIPS):
        vecs, meta = [], []
        for p in todo[g:g+SHARD_CLIPS]:
            try:
                frames, ts = read_frames(p)
                if not frames: continue
                vecs.append(embed_images(frames))
                meta += [{"clip": stem(p), "t": t} for t in ts]
            except Exception as e:
                print("skip", p, repr(e)[:100])
        if not vecs: continue
        np.save(f"{OUT}/shards/s{shard_id:05d}.npy", np.vstack(vecs).astype(np.float16))
        with open(f"{OUT}/shards/s{shard_id:05d}.jsonl", "w") as f:   # written last = shard complete
            for m in meta: f.write(json.dumps(m) + "\n")
        shard_id += 1
        print(f"{min(g+SHARD_CLIPS, len(todo))}/{len(todo)} clips, {time.time()-t0:.0f}s", flush=True)

def finalize():
    arrs, meta = [], []
    for p in sorted(glob.glob(f"{OUT}/shards/*.npy")):
        arrs.append(np.load(p)); meta += [json.loads(l) for l in open(p[:-4] + ".jsonl")]
    X = np.vstack(arrs).astype("float32")
    assert len(X) == len(meta)
    n, d = X.shape
    with open(f"{OUT}/meta.jsonl", "w") as f:
        for m in meta: f.write(json.dumps(m) + "\n")

    flat = faiss.IndexFlatIP(d); flat.add(X)
    sq = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    sq.train(X); sq.add(X)
    idxs = {"flat": flat, "sq8": sq}
    if n >= 2560:                                   # PQ needs enough training vectors
        pq = faiss.IndexPQ(d, 64, 8, faiss.METRIC_INNER_PRODUCT)
        pq.train(X); pq.add(X); idxs["pq64"] = pq
    for k, ix in idxs.items():
        faiss.write_index(ix, f"{OUT}/{k}.faiss")
        print(f"{k:6s} {os.path.getsize(f'{OUT}/{k}.faiss')/1e6:8.1f} MB")
    print(f"vectors: {n}, dim: {d}, clips: {len({m['clip'] for m in meta})}")

    rng = np.random.default_rng(0)                  # how much do compressed indexes agree with flat?
    Q = X[rng.choice(n, min(200, n), replace=False)]
    _, Ig = flat.search(Q, 10)
    for k, ix in idxs.items():
        if k == "flat": continue
        _, Ic = ix.search(Q, 10)
        print(f"{k} overlap@10 vs flat: {np.mean([len(set(a) & set(b))/10 for a, b in zip(Ig, Ic)]):.3f}")

def search(q, k=10, which="sq8"):
    ix = faiss.read_index(f"{OUT}/{which}.faiss")
    meta = [json.loads(l) for l in open(f"{OUT}/meta.jsonl")]
    t = time.perf_counter()
    D, I = ix.search(embed_text([q]), 300)
    print(f"search {1000*(time.perf_counter()-t):.1f} ms")
    best = {}
    for s, i in zip(D[0], I[0]):
        if i >= 0 and meta[i]["clip"] not in best:   # results are sorted, first hit = best frame
            best[meta[i]["clip"]] = (float(s), meta[i]["t"])
    for c, (s, t_) in list(best.items())[:k]:
        print(f"{s:.3f}  {c}  t={t_}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "finalize", "search"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--index", default="sq8")
    a = ap.parse_args()
    if a.cmd == "build": build(a.n)
    elif a.cmd == "finalize": finalize()
    else: search(a.arg, which=a.index)