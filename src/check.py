import faiss
faiss.omp_set_num_threads(1)          # import faiss before torch
import json, os, sys
from clipindex import embed_text, OUT

query = sys.argv[1]                   # e.g. "a rainy night road"
field, value = sys.argv[2], sys.argv[3]   # e.g. weather rainy

# labels = {os.path.splitext(x["name"])[0]: x for x in json.load(open("labels_slim.json"))}
import glob
import json
import os

labels = {
    os.path.splitext(x["name"])[0]: x
    for path in glob.glob("/Users/exodussubhash/Downloads/100k/train/*.json")
    for x in [json.load(open(path))]
}
# print(next(iter(labels.items())))
meta = [json.loads(l) for l in open(f"{OUT}/meta.jsonl")]
index = faiss.read_index(f"{OUT}/flat.faiss")

D, I = index.search(embed_text([query]), 100)
seen = []
for i in I[0]:
    c = meta[i]["clip"]
    if c not in seen:
        seen.append(c)
top = seen[:10]

good = 0
for c in top:
    ok = labels[c]['attributes'][field] == value
    good += ok
    print("OK " if ok else "NO ", c, labels[c]['attributes'][field])

all_clips = {m["clip"] for m in meta}
base = sum(labels[c]['attributes'][field] == value for c in all_clips) / len(all_clips)
print(f"\nprecision@10 = {good/10:.2f}   random baseline = {base:.2f}")

