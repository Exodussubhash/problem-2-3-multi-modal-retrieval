# main.py - Dashcam multimodal retrieval: text query -> CLIP -> FAISS, scored against BDD100K labels
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"      # Mac: faiss + torch both ship libomp
os.environ["OMP_NUM_THREADS"] = "1"
import faiss                                      # import faiss before torch
faiss.omp_set_num_threads(1)

import argparse, json, sys, time
from pathlib import Path
from config import CONFIG, resolve_path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import clipindex as ci                            # CLIP + indexing code

# ---- evaluation queries: (text query, rule over a clip's BDD100K label dict) ----
def attr(L, f):
    return (L.get("attributes") or {}).get(f)

def has_obj(L, cat):                              # per-clip files list objects under "frames"
    return any(o.get("category") == cat
               for fr in L.get("frames", []) for o in fr.get("objects", []))

QUERIES = [
    ("a rainy road",              lambda L: attr(L, "weather") == "rainy"),
    ("a snowy road",              lambda L: attr(L, "weather") == "snowy"),
    ("a foggy road",              lambda L: attr(L, "weather") == "foggy"),
    ("night driving",             lambda L: attr(L, "timeofday") == "night"),
    ("a highway",                 lambda L: attr(L, "scene") == "highway"),
    ("a tunnel",                  lambda L: attr(L, "scene") == "tunnel"),
    ("a residential street",      lambda L: attr(L, "scene") == "residential"),
    ("a rainy night road",        lambda L: attr(L, "weather") == "rainy" and attr(L, "timeofday") == "night"),
    ("a pedestrian on the road",  lambda L: has_obj(L, "person")),
    ("a traffic light at night",  lambda L: has_obj(L, "traffic light") and attr(L, "timeofday") == "night"),
    ("a truck on the road",       lambda L: has_obj(L, "truck")),
]

def load_labels(labels_dir, clips):
    out = {}
    for c in clips:                               # read only the label files we need
        p = os.path.join(labels_dir, f"{c}.json")
        if os.path.exists(p):
            out[c] = json.load(open(p))
    return out

def run_eval(labels_dir, indexes, only10, min_rel):
    meta = [json.loads(l) for l in open(f"{ci.OUT}/meta.jsonl")]
    indexed = sorted({m["clip"] for m in meta})
    labels = load_labels(labels_dir, indexed)
    clips = sorted(labels)
    print(f"indexed clips: {len(indexed)} | with labels: {len(clips)} | vectors: {len(meta)}")
    if not clips:
        sys.exit("no label files matched the indexed clips - check --labels-dir and file names")
    results = []
    for which in indexes:
        path = f"{ci.OUT}/{which}.faiss"
        if not os.path.exists(path):
            continue
        index = faiss.read_index(path)
        mb = os.path.getsize(path) / 1e6
        print(f"\n=== {which}: {mb:.1f} MB ===")
        print(f"{'query':26s}{'rel':>5s}{'base':>7s}{'P@10':>7s}{'R@50':>7s}{'lift':>7s}{'ms':>8s}")
        for q, rule in QUERIES:
            rel = {c for c in clips if rule(labels[c])}
            if len(rel) < min_rel:
                print(f"{q:26s}{len(rel):5d}   skipped (<{min_rel} relevant clips)")
                continue
            qv = ci.embed_text([q])
            t = time.perf_counter()
            _, I = index.search(qv, min(3000, index.ntotal))
            ms = 1000 * (time.perf_counter() - t)
            seen = []
            for i in I[0]:
                if i < 0: continue
                m = meta[i]
                if only10 and m["t"] != 10: continue       # the labeled frame only
                if m["clip"] in labels and m["clip"] not in seen:
                    seen.append(m["clip"])
            hit = [c in rel for c in seen]
            p10 = sum(hit[:10]) / max(len(hit[:10]), 1)
            r50 = sum(hit[:50]) / len(rel)
            base = len(rel) / len(clips)
            results.append({"index": which, "query": q, "relevant": len(rel), "base_rate": round(base, 3),
                            "P@10": round(p10, 3), "R@50": round(r50, 3),
                            "lift": round(p10 / base, 2), "search_ms": round(ms, 1), "index_mb": round(mb, 2)})
            print(f"{q:26s}{len(rel):5d}{base:7.2f}{p10:7.2f}{r50:7.2f}{p10/base:7.1f}{ms:8.1f}")
    os.makedirs(ROOT / "results", exist_ok=True)
    json.dump(results, open(ROOT / "results" / "eval.json", "w"), indent=1)
    print("\nsummary (mean over scored queries):")
    for which in indexes:
        r = [x for x in results if x["index"] == which]
        if r:
            print(f"  {which:6s} P@10={sum(x['P@10'] for x in r)/len(r):.3f}  "
                  f"lift={sum(x['lift'] for x in r)/len(r):.1f}x  "
                  f"search={sum(x['search_ms'] for x in r)/len(r):.1f} ms  size={r[0]['index_mb']} MB  (n={len(r)})")

def run_build(video_dir, n):
    ci.VID_DIR = video_dir
    t0 = time.perf_counter(); ci.build(n)
    t1 = time.perf_counter(); ci.finalize()
    t2 = time.perf_counter()
    meta = [json.loads(l) for l in open(f"{ci.OUT}/meta.jsonl")]
    clips = len({m["clip"] for m in meta})
    print(f"\nembedding: {t1-t0:.0f}s ({clips/max(t1-t0,1e-9):.2f} clips/s in this run, includes resume skips) | "
          f"index build: {t2-t1:.0f}s | clips: {clips} | vectors: {len(meta)}")

def run_demo(video_dir, n):
    ci.OUT = str(ROOT / "index_demo")             # keep demo output separate
    run_build(video_dir, n)
    for q in ["a rainy night road", "a highway in daylight", "a pedestrian crossing the street"]:
        print(f"\nquery: {q}")
        ci.search(q, k=5, which="flat")
        

def main():
    p = argparse.ArgumentParser(description="Dashcam multimodal retrieval (CLIP + FAISS) scored on BDD100K labels")
    p.add_argument("--mode", choices=["demo", "build", "search", "eval", "all"], default="demo")
    p.add_argument("--video-dir", default=os.environ.get(
        "BDD_VIDEO_DIR", str(resolve_path(CONFIG["data"]["video_dir"]))))
    p.add_argument("--labels-dir", default=os.environ.get(
        "BDD_LABELS_DIR", str(resolve_path(CONFIG["data"]["labels_dir"]))),
                   help="folder of per-clip BDD100K label .json files")
    p.add_argument("--n", type=int, default=20, help="max clips to index")
    p.add_argument("--query", default="a rainy night road")
    p.add_argument("--index", nargs="+", default=["flat", "sq8", "pq64"])
    p.add_argument("--only10", action="store_true", help="score only the frame at t=10s (the labeled frame)")
    p.add_argument("--min-rel", type=int, default=5)
    a = p.parse_args()

    if a.mode == "demo":
        run_demo(a.video_dir, a.n); return
    if a.mode in ("build", "all"):
        run_build(a.video_dir, a.n)
    if a.mode == "search":
        ci.search(a.query, k=10, which=a.index[0])
    if a.mode in ("eval", "all"):
        if not a.labels_dir:
            sys.exit("eval needs --labels-dir (or BDD_LABELS_DIR)")
        run_eval(a.labels_dir, a.index, a.only10, a.min_rel)

if __name__ == "__main__":
    main()
