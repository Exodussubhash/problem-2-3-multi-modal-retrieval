"""Search the FAISS index and report labeled precision for the retrieved clips."""

import argparse
import glob
import json
import os
from pathlib import Path

from config import CONFIG, resolve_path
from retrieval_eval import label_prevalence, precision_at_k, unique_clips_from_hits


LABEL_DIR = resolve_path(CONFIG["data"]["labels_dir"])


def load_labels(directory=LABEL_DIR):
    labels = {}
    for path in glob.glob(str(Path(directory) / "*.json")):
        with open(path, encoding="utf-8") as file:
            item = json.load(file)
        labels[os.path.splitext(item["name"])[0]] = item
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Text query, for example: 'a rainy night road'")
    parser.add_argument("field", help="Label field to evaluate, for example: weather")
    parser.add_argument("value", help="Expected label value, for example: rainy")
    args = parser.parse_args()

    import faiss
    faiss.omp_set_num_threads(1)  # import FAISS before clipindex imports torch
    from clipindex import OUT, embed_text

    labels = load_labels()
    with open(f"{OUT}/meta.jsonl", encoding="utf-8") as file:
        meta = [json.loads(line) for line in file]
    index = faiss.read_index(f"{OUT}/flat.faiss")

    _, indices = index.search(embed_text([args.query]), 200)
    top = unique_clips_from_hits(indices[0], meta, limit=10)
    for clip in top:
        actual = labels[clip]["attributes"][args.field]
        print(("OK  " if actual == args.value else "NO  ") + f"{clip} {actual}")

    indexed_clips = {item["clip"] for item in meta}
    precision = precision_at_k(top, labels, args.field, args.value)
    baseline = label_prevalence(indexed_clips, labels, args.field, args.value)
    print(f"\nprecision@{len(top)} = {precision:.2f}   random baseline = {baseline:.2f}")


if __name__ == "__main__":
    main()
