#!/usr/bin/env python3
"""
Merge initial and finetuned WER result files for side-by-side comparison.
Aligned by utt ID; ref is assumed identical across both files.
"""

import re
import sys


def parse_wer_file(path):
    """Parse a WER result file into a dict keyed by utt ID."""
    entries = {}
    summary_lines = []
    in_summary = True

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Collect header summary (everything before the first "utt:" line)
        if in_summary:
            if line.startswith("utt:"):
                in_summary = False
            else:
                summary_lines.append(line)
                i += 1
                continue

        if line.startswith("utt:"):
            utt_id = line[len("utt:"):].strip()
            wer_line = lines[i + 1].rstrip("\n") if i + 1 < len(lines) else ""
            ref_line = lines[i + 2].rstrip("\n") if i + 2 < len(lines) else ""
            hyp_line = lines[i + 3].rstrip("\n") if i + 3 < len(lines) else ""
            # Strip "ref:" / "hyp:" prefix and leading whitespace
            ref_text = re.sub(r"^\s*ref:\s*", "", ref_line)
            hyp_text = re.sub(r"^\s*hyp:\s*", "", hyp_line)
            entries[utt_id] = {
                "wer_line": wer_line.strip(),
                "ref_text": ref_text,
                "hyp_text": hyp_text,
            }
            i += 4
        else:
            i += 1

    return summary_lines, entries


def extract_wer_value(wer_line):
    """Extract WER percentage from a WER line, e.g. 'WER: 18.63 % ...'"""
    m = re.search(r"WER:\s*([\d.]+)\s*%", wer_line)
    return m.group(1) if m else "N/A"


def main():
    initial_path = "results/initial_test_wer.txt"
    finetuned_path = "results/finetuned_test_wer.txt"
    output_path = "results/merged_comparison.txt"

    if len(sys.argv) == 3:
        initial_path, finetuned_path = sys.argv[1], sys.argv[2]

    initial_summary, initial_entries = parse_wer_file(initial_path)
    finetuned_summary, finetuned_entries = parse_wer_file(finetuned_path)

    all_utts = set(initial_entries) | set(finetuned_entries)

    def sort_key(utt):
        if utt in initial_entries and utt in finetuned_entries:
            try:
                return float(extract_wer_value(finetuned_entries[utt]["wer_line"])) - \
                       float(extract_wer_value(initial_entries[utt]["wer_line"]))
            except ValueError:
                pass
        return float("inf")  # utts missing from one file go to the end

    all_utts = sorted(all_utts, key=sort_key, reverse=True)

    with open(output_path, "w", encoding="utf-8") as out:
        # Write both summaries
        out.write("=" * 79 + "\n")
        out.write("INITIAL MODEL SUMMARY\n")
        out.write("=" * 79 + "\n")
        for line in initial_summary:
            out.write(line + "\n")

        out.write("\n" + "=" * 79 + "\n")
        out.write("FINETUNED MODEL SUMMARY\n")
        out.write("=" * 79 + "\n")
        for line in finetuned_summary:
            out.write(line + "\n")

        out.write("\n" + "=" * 79 + "\n")
        out.write(f"PER-UTTERANCE COMPARISON  ({len(all_utts)} utterances)\n")
        out.write("=" * 79 + "\n\n")

        only_initial = 0
        only_finetuned = 0
        both = 0

        for utt in all_utts:
            has_initial = utt in initial_entries
            has_finetuned = utt in finetuned_entries

            if has_initial and has_finetuned:
                both += 1
                ini = initial_entries[utt]
                fin = finetuned_entries[utt]
                wer_ini = extract_wer_value(ini["wer_line"])
                wer_fin = extract_wer_value(fin["wer_line"])

                try:
                    delta = float(wer_fin) - float(wer_ini)
                    delta_str = f"{delta:+.2f}%"
                except ValueError:
                    delta_str = "N/A"

                out.write(f"utt: {utt}\n")
                out.write(f"WER  initial : {wer_ini}%   ({ini['wer_line']})\n")
                out.write(f"WER finetuned: {wer_fin}%   ({fin['wer_line']})   delta={delta_str}\n")
                out.write(f"ref          : {ini['ref_text']}\n")
                out.write(f"hyp  initial : {ini['hyp_text']}\n")
                out.write(f"hyp finetuned: {fin['hyp_text']}\n")

            elif has_initial:
                only_initial += 1
                ini = initial_entries[utt]
                out.write(f"utt: {utt}  [initial only]\n")
                out.write(f"WER  initial : {extract_wer_value(ini['wer_line'])}%   ({ini['wer_line']})\n")
                out.write(f"ref          : {ini['ref_text']}\n")
                out.write(f"hyp  initial : {ini['hyp_text']}\n")
                out.write(f"hyp finetuned: (not present)\n")

            else:
                only_finetuned += 1
                fin = finetuned_entries[utt]
                out.write(f"utt: {utt}  [finetuned only]\n")
                out.write(f"WER finetuned: {extract_wer_value(fin['wer_line'])}%   ({fin['wer_line']})\n")
                out.write(f"ref          : {fin['ref_text']}\n")
                out.write(f"hyp  initial : (not present)\n")
                out.write(f"hyp finetuned: {fin['hyp_text']}\n")

            out.write("\n")

        out.write("=" * 79 + "\n")
        out.write(f"Coverage: both={both}  initial_only={only_initial}  finetuned_only={only_finetuned}\n")
        out.write("=" * 79 + "\n")

    print(f"Merged {len(all_utts)} utterances -> {output_path}")
    print(f"  both={both}, initial_only={only_initial}, finetuned_only={only_finetuned}")


if __name__ == "__main__":
    main()