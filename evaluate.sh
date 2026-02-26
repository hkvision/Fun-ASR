export PYTHONPATH=/home/arda/kai/Fun-ASR-Nano-2512/compute-weighted-wer/:$PYTHONPATH

# MODEL=/home/arda/kai/Fun-ASR-Nano-2512/
MODEL=/home/arda/kai/Fun-ASR-Nano-2512/Fun-ASR/outputs

DATA=audios/dataset/test_wav.scp
GROUNDTRUTH=audios/dataset/test_text_norm.txt
# DATA=audios/dataset/train_val_wav.scp
# GROUNDTRUTH=audios/dataset/train_val_text_norm.txt
# DATA=audios/dataset/different_humans_wav.scp
# GROUNDTRUTH=audios/dataset/different_humans_text_norm.txt

# OUTPUT=audios/results/initial_test_wer.txt
# OUTPUT=audios/results/initial_train_val_wer.txt
# OUTPUT=audios/results/initial_different_humans_wer.txt
OUTPUT=audios/results/finetuned_test_wer.txt
# OUTPUT=audios/results/finetuned_train_val_wer.txt
# OUTPUT=audios/results/finetuned_different_humans_wer.txt

python decode.py \
  ++model_dir=${MODEL} \
  ++scp_file=${DATA} \
  ++output_file=predictions.txt

python tools/whisper_mix_normalize.py predictions.txt predictions_norm.txt

python /home/arda/kai/Fun-ASR-Nano-2512/compute-weighted-wer/compute_wer/cli.py ${GROUNDTRUTH} predictions_norm.txt audios/results/wer.txt --hotword-file audios/hotwords.txt
# without hotword
# python /home/arda/kai/Fun-ASR-Nano-2512/compute-weighted-wer/compute_wer/cli.py ${GROUNDTRUTH} predictions_norm.txt audios/results/wer.txt

python audios/sort_wer.py

rm audios/results/wer.txt
mv audios/results/sorted_wer.txt ${OUTPUT}

rm predictions*.txt
