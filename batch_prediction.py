from funasr import AutoModel

# model_dir = "/home/arda/kai/Fun-ASR-Nano-2512"
model_dir = "/home/arda/kai/Fun-ASR-Nano-2512/Fun-ASR/experiments/finetune/data_with_S/outputs"
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    # device="cpu",
    # vad_model="fsmn-vad",
    # vad_kwargs={"max_single_segment_time": 30000},
)

audio_dir = "Fun-ASR/audios/02_03_17_54/"
import os
audios = os.listdir(audio_dir)
audios = [audio for audio in audios if not audio.endswith("txt")]
output_dir = "Fun-ASR/audios/02_03_17_54_finetuned/"

with open("audios/hotwords.txt", "r", encoding="utf-8") as f:
    hotwords = [line.strip() for line in f if line.strip()]

for audio in audios:
    wav_path = audio_dir + audio
    res = model.generate(
        input=[wav_path],
        cache={},
        batch_size=1,
        hotwords=hotwords,
        # 中文、英文、日文 for Fun-ASR-Nano-2512
        # 中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
        # 印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
        # 匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
        # 斯洛伐克语、斯洛文尼亚语、瑞典语 for Fun-ASR-MLT-Nano-2512
        # language="中文",
        # itn=True, # or False
    )
    text = res[0]["text"]
    print(text)
    with open(output_dir + audio.replace("wav", "txt"), 'w') as f:
        f.write(text)
