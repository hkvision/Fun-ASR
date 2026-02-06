python decode.py \
  ++model_dir=/home/arda/kai/Fun-ASR-Nano-2512/Fun-ASR/outputs \
  ++scp_file=experiments/finetune/val_wav.scp \
  ++output_file=experiments/prediction_val_with_hotword_finetuned.txt

python tools/whisper_mix_normalize.py experiments/prediction_val_with_hotword_finetuned.txt experiments/prediction_val_with_hotword_finetuned_norm.txt

compute-wer experiments/ground_truth_val_norm.txt experiments/prediction_val_with_hotword_finetuned_norm.txt experiments/finetuned_val_wer_with_hotword.txt
