#!/usr/bin/env python
"""Convert the local HF MRPC JSONL files to the original SentEval TSV format."""

import argparse
import json
from pathlib import Path


def convert(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        fout.write("label\tid1\tid2\tsentence1\tsentence2\n")
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            fout.write("{label}\t{idx}\t{idx}\t{s1}\t{s2}\n".format(
                label=int(row["label"]), idx=int(row.get("idx", n)),
                s1=row["text1"], s2=row["text2"]))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/zsx/nlp/data/senteval/downstream/MRPC"))
    args = ap.parse_args()
    counts = {
        "train": convert(args.root / "train.jsonl", args.root / "msr_paraphrase_train.txt"),
        "test": convert(args.root / "test.jsonl", args.root / "msr_paraphrase_test.txt"),
    }
    print(counts)


if __name__ == "__main__":
    main()
