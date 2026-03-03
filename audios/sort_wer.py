import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "results/wer.txt")
output_file = os.path.join(script_dir, "results/sorted_wer.txt")

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Split off the trailing summary block (starts with ===)
summary_match = re.search(r"(={3,}.*)", content, re.DOTALL)
summary_block = summary_match.group(1).strip() if summary_match else ""
body = content[:summary_match.start()] if summary_match else content

# Parse individual entries (separated by blank lines)
raw_entries = [e.strip() for e in re.split(r"\n{2,}", body) if e.strip()]

entries = []
for entry in raw_entries:
    wer_match = re.search(r"WER:\s*([\d.]+)\s*%", entry)
    wer = float(wer_match.group(1)) if wer_match else 0.0
    entries.append((wer, entry))

# Sort by WER descending
entries.sort(key=lambda x: x[0], reverse=True)

with open(output_file, "w", encoding="utf-8") as f:
    if summary_block:
        f.write(summary_block + "\n\n")
    for _, entry in entries:
        f.write(entry + "\n\n")

print(f"Done: {len(entries)} entries sorted by WER descending.")
print(f"Output: {output_file}")