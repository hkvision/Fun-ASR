import os
import json
from funasr import AutoModel

model_dir = "/home/arda/kai/LenovoSmartMeeting/Fun-ASR-Nano-2512"
# model_dir = "/home/arda/kai/LenovoSmartMeeting/Fun-ASR/outputs"
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    # device="cpu",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
)
# Disable dither for deterministic inference (dither is for training only)
model.kwargs["frontend"].dither = 0.0
if hasattr(model, "_base_kwargs_map") and "kwargs" in model._base_kwargs_map:
    model._base_kwargs_map["kwargs"]["frontend"].dither = 0.0

# with open("audios/hotwords.txt", "r", encoding="utf-8") as f:
#     hotwords = [line.strip() for line in f if line.strip()]
with open("audios/dataset/keyword_hits.json", encoding="utf-8") as f:
    keyword_hits = json.load(f)

audio_dir = "audios/real_meeting/PantherLake/"
audios = os.listdir(audio_dir)
audios = [audio for audio in audios if not audio.endswith("txt")]
audios = ["/home/arda/kai/LenovoSmartMeeting/Fun-ASR/" + audio_dir + audio for audio in audios]
# audios = [audios[1], audios[2], audios[0]]
# audios = [audios[0]]
# hotwords = ["PantherLake"]

# wav_path = "/home/arda/kai/LenovoSmartMeeting/Fun-ASR/audios/Intel_hotword_99/tts/XE/10_v1_uncle_fu_mature.wav"
# audios = [wav_path]
results = []
for audio in audios:
    # print(audio)
    audio_utt = "PantherLake_real_" + audio.split("/")[-1].split(".")[0]
    hotwords = keyword_hits[audio_utt]
    print("{} hotwords: {}".format(audio_utt, hotwords))
    res = model.generate(
        input=[audio],
        cache={},
        batch_size=1,
        hotwords=hotwords,
        # hotwords=[],
    )
    text = res[0]["text"]
    # print(text)
    results.append((audio_utt, text))

for result in results:
    print("{} {}".format(result[0], result[1]))

# model = AutoModel(
#     model=model_dir,
#     trust_remote_code=True,
#     vad_model="fsmn-vad",
#     vad_kwargs={"max_single_segment_time": 30000},
#     remote_code="./model.py",
#     device="cuda:0",
# )
# res = model.generate(input=[wav_path], cache={}, batch_size=1)
# text = res[0]["text"]
# print(text)
