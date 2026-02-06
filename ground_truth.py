import os

all_audios = []
parent_dir = "audios"
subfolders = os.listdir(parent_dir)
subfolders = [os.path.join(parent_dir, folder) for folder in subfolders]

for folder in subfolders:
    audios = os.listdir(folder)
    audios = [os.path.join(folder, audio) for audio in audios]
    audios = [audio for audio in audios if audio.endswith(".txt")]
    all_audios += audios

with open("ground_truth_all_S.txt", 'w') as file:
    for audio in all_audios:
        key = audio.split("/")[-1].replace(".txt", "")
        with open(audio, 'r') as label_file:
            label = label_file.read()
        file.write(f"{key}\t{label}\n")
print("Succeed")
