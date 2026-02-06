import os
from funasr import AutoModel

model_dir = "/home/arda/kai/Fun-ASR-Nano-2512"
# model_dir = "/home/arda/kai/Fun-ASR-Nano-2512/Fun-ASR/experiments/finetune/lr0.0002-4epochs-llmonly/outputs"
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    # device="cpu",
    # vad_model="fsmn-vad",
    # vad_kwargs={"max_single_segment_time": 30000},
)

audio_dir = "Fun-ASR/audios/samples_S/"
audios = os.listdir(audio_dir)
audios = [audio for audio in audios if not audio.endswith("txt")]
# audios = [audios[1]]

# wav_path = "/home/arda/kai/Fun-ASR-Nano-2512/Fun-ASR/error/error_铁威马/seg_12_24_15_04_720.wav"
wav_path = "/home/arda/kai/Fun-ASR-Nano-2512/Fun-ASR/error/error_heliconsearch/seg_12_24_15_04_499.wav"
audios = [wav_path]
results = []
for audio in audios:
    # wav_path = audio_dir + audio
    res = model.generate(
        input=[wav_path],
        cache={},
        batch_size=1,
        hotwords=["千问", "Xe", "Xe Core", "Lunar Lake", "Panther Lake", "Helicon Search", "Arrow Lake", "Helicon Search", "极空间", "铁威马"],
        # 中文、英文、日文 for Fun-ASR-Nano-2512
        # 中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
        # 印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
        # 匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
        # 斯洛伐克语、斯洛文尼亚语、瑞典语 for Fun-ASR-MLT-Nano-2512
        # language="中文",
        # itn=True, # or False
    )
    text = res[0]["text"]
    # print(text)
    results.append((audio, text))

for result in results:
    print("=============", result[0])
    print(result[1])

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
