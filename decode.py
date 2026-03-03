import os
import json

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
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        remote_code="./model.py",
        device=device,
    )
    # Disable dither for deterministic inference (dither is for training only)
    model.kwargs["frontend"].dither = 0.0
    if hasattr(model, "_base_kwargs_map") and "kwargs" in model._base_kwargs_map:
        model._base_kwargs_map["kwargs"]["frontend"].dither = 0.0
    
    # with open("audios/hotwords.txt", "r", encoding="utf-8") as f:
    #     hotwords = [line.strip() for line in f if line.strip()]
    # print("hotwords: ", hotwords)
    with open("audios/dataset/keyword_hits.json", encoding="utf-8") as f:
        keyword_hits = json.load(f)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(scp_file, "r", encoding="utf-8") as f1:
        with open(output_file, "w", encoding="utf-8") as f2:
            for line in f1:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                if len(parts) == 2:
                    hotwords = keyword_hits[parts[0]]
                    print("{} hotwords: {}".format(parts[0], hotwords))
                    # text = model.generate(input=[parts[1]], cache={}, batch_size=1)[0]["text"]
                    text = model.generate(input=[parts[1]], cache={}, batch_size=1, hotwords=hotwords)[0]["text"]
                    f2.write(f"{parts[0]}\t{text}\n")


if __name__ == "__main__":
    main_hydra()
