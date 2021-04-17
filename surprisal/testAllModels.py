# Running  all language models on the test set

from config import CHECKPOINT_HOME
import os
import subprocess

models = [x for x in os.listdir(CHECKPOINT_HOME) if x.endswith(".pth.tar")]
for model in models:
   model = model.replace(".pth.tar", "")
   command = f"python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-TEST.py --language dailymail --load-from {model} --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --layer_num 3 --learning_rate 3.6 --lr_decay 0.95 --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0".split(" ")
   print(command)
   subprocess.call(command)
