import os

all_audios = []
parent_dir = "audios"
subfolders = os.listdir(parent_dir)
subfolders = [os.path.join(parent_dir, folder) for folder in subfolders]

for folder in subfolders:
    audios = os.listdir(folder)
    audios = [os.path.join(folder, audio) for audio in audios]
    audios = [audio for audio in audios if audio.endswith(".wav") or audio.endswith(".mp3")]
    all_audios += audios

with open("all_audios_S.scp", 'w') as file:
    for audio in all_audios:
        key = audio.split("/")[-1].replace(".wav", "").replace(".mp3", "")
        path = os.path.abspath(audio)
        file.write(f"{key}\t{path}\n")
print("Succeed")
