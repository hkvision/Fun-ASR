import os

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item
    kwargs = to_plain_list(cfg)

    model_dir = kwargs.get("model_dir", "FunAudioLLM/Fun-ASR-Nano-2512")
    scp_file = kwargs["scp_file"]
    output_file = kwargs["output_file"]

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    from funasr import AutoModel

    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        # vad_model="fsmn-vad",
        # vad_kwargs={"max_single_segment_time": 30000},
        remote_code="./model.py",
        device=device,
    )
    
    original_model = AutoModel(
        model="/home/arda/kai/Fun-ASR-Nano-2512",
        trust_remote_code=True,
        # vad_model="fsmn-vad",
        # vad_kwargs={"max_single_segment_time": 30000},
        remote_code="./model.py",
        device=device,
    )

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    audios = []
    original_outputs = []
    original_outputs_with_hotword = []
    finetuned_outputs_with_hotword = []
    with open(scp_file, "r", encoding="utf-8") as f1:
        for line in f1:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                audios.append((parts[0], parts[1]))
    for _, audio in audios:
        text = original_model.generate(input=[audio], cache={}, batch_size=1)[0]["text"]
        original_outputs.append(text)
    for _, audio in audios:
        text = original_model.generate(input=[audio], cache={}, batch_size=1, hotwords=["千问", "Xe", "Xe Core", "Lunar Lake", "Panther Lake", "Helicon Search", "Arrow Lake", "Helicon Search", "极空间", "铁威马"])[0]["text"]
        original_outputs_with_hotword.append(text)
    for _, audio in audios:
        text = model.generate(input=[audio], cache={}, batch_size=1, hotwords=["千问", "Xe", "Xe Core", "Lunar Lake", "Panther Lake", "Helicon Search", "Arrow Lake", "Helicon Search", "极空间", "铁威马"])[0]["text"]
        finetuned_outputs_with_hotword.append(text)
    with open(output_file, "w", encoding="utf-8") as f2:
        for i, audio in enumerate(audios):
            f2.write(f"{audio[0]}\n")
            f2.write(f"{original_outputs[i]}\n")
            f2.write(f"{original_outputs_with_hotword[i]}\n")
            f2.write(f"{finetuned_outputs_with_hotword[i]}\n")
            f2.write(f"\n")


if __name__ == "__main__":
    main_hydra()
