import os

script_dir = os.path.dirname(os.path.abspath(__file__))

scp_files = ["dataset/train_wav.scp", "dataset/val_wav.scp"]
txt_files = ["dataset/train_text.txt", "dataset/val_text.txt"]

def merge_files(filenames, output_name):
    output_path = os.path.join(script_dir, output_name)
    total_lines = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for fname in filenames:
            fpath = os.path.join(script_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                out.writelines(lines)
                total_lines += len(lines)
            print(f"  {fname}: {len(lines)} lines")
    print(f"=> {output_name}: {total_lines} lines total\n")

print("Merging .scp files...")
merge_files(scp_files, "dataset/train_val_wav.scp")

print("Merging .txt files...")
merge_files(txt_files, "dataset/train_val_text.txt")