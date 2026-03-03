"""
从 audios/dataset/ 下不含 norm 的 txt 文件中，
提取包含 hotwords.txt 里关键词的句子，结果存为 JSON。

输出格式（每个含关键词的条目）：
[
  {"id": "...", "text": "...", "keywords": ["kw1", "kw2"]},
  ...
]
"""

import json
import os
import glob
import argparse


def load_hotwords(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_keywords(text, hotwords):
    return [kw for kw in hotwords if kw in text]


def process_txt_file(path, hotwords):
    results = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if "\t" in line:
                utt_id, text = line.split("\t", 1)
            else:
                utt_id = f"{os.path.basename(path)}:L{lineno}"
                text = line
            matched = find_keywords(text, hotwords)
            if matched:
                results.append({"id": utt_id, "text": text, "keywords": matched})
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract hotword-containing utterances.")
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory containing the txt files",
    )
    parser.add_argument(
        "--hotwords",
        default=os.path.join(os.path.dirname(__file__), "hotwords.txt"),
        help="Path to hotwords.txt",
    )
    parser.add_argument(
        "--output",
        default="keyword_hits.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    hotwords = load_hotwords(args.hotwords)
    print(f"Loaded {len(hotwords)} hotwords: {hotwords}")

    txt_files = [
        p for p in glob.glob(os.path.join(args.dataset_dir, "*.txt"))
        if "norm" not in os.path.basename(p)
    ]
    print(f"Found {len(txt_files)} txt files (without norm):")
    for p in sorted(txt_files):
        print(f"  {os.path.basename(p)}")

    # 先处理非合并文件，再处理合并文件，保证去重时优先保留原始来源
    def sort_key(p):
        name = os.path.basename(p)
        # train_val_text.txt 排最后，确保被去重时丢弃
        return (1 if "train_val" in name else 0, name)

    all_results = {}
    for path in sorted(txt_files, key=sort_key):
        hits = process_txt_file(path, hotwords)
        fname = os.path.basename(path)
        kept = 0
        for item in hits:
            if item["id"] not in all_results:
                all_results[item["id"]] = item["keywords"]
                kept += 1
        skipped = len(hits) - kept
        print(f"  {fname}: {len(hits)} hits, kept {kept}, skipped {skipped} (duplicates)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nTotal {len(all_results)} unique hits -> {args.output}")


if __name__ == "__main__":
    main()