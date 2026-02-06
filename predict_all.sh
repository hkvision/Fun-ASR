python decode.py \
  ++model_dir=/home/arda/kai/Fun-ASR-Nano-2512/Fun-ASR/outputs \
  ++scp_file=experiments/all_audios.scp \
  ++output_file=experiments/prediction_all_with_hotword_finetuned.txt

python tools/whisper_mix_normalize.py experiments/prediction_all_with_hotword_finetuned.txt experiments/prediction_all_with_hotword_finetuned_norm.txt

compute-wer experiments/ground_truth_all_norm.txt experiments/prediction_all_with_hotword_finetuned_norm.txt experiments/finetuned_wer_with_hotword.txt
