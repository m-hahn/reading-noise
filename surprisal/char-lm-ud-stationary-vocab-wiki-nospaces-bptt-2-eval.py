import sys


# python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-eval.py --batchSize 1 --char_embedding_size 200 --hidden_dim 1024 --language dailymail --layer_num 3 --load-from test-661158218 --verbose True


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([128, 128, 128, 128, 128, 128, 128, 256, 512]))
parser.add_argument("--char_embedding_size", type=int, default=random.choice([100, 100, 100, 200, 200, 200, 200]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([2,3]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0, 0.0,  0.01, 0.05, 0.1]))
parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0, 0.0, 0.0,  0.01, 0.02,0.03, 0.0, 0.0, 0.0,  0.01, 0.02,0.03,  0.05, 0.1,  0.15, 0.2]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.01, 0.01]))
parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([2.0, 2.2, 2.4, 2.6, 2.8, 2.9, 3.0, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5,3.6, 3.7, 3.8, 3.9, 4.0, 3.1, 3.2, 3.3, 3.4, 3.5,3.6, 3.7, 3.8, 3.9, 4.0, 3.1, 3.2, 3.3, 3.4, 3.5,3.6, 3.7, 3.8, 3.9]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50, 50, 50, 80]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([0.5, 0.7, 0.9, 0.95, 0.98, 0.98, 0.98, 0.98, 1.0]))


import math

args=parser.parse_args()
assert args.batchSize == 1
#if "MYID" in args.save_to:
#   args.save_to = args.save_to.replace("MYID", str(args.myID))

print(args)



import corpusIteratorWiki



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

from config import CHAR_VOCAB_HOME

try:
   with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language+".txt", "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
    assert False
    print("Creating new vocab")
    char_counts = {}
    # get symbol vocabulary

    with open("/private/home/mhahn/data/WIKIPEDIA/"+args.language+"-vocab.txt", "r") as inFile:
      words = inFile.read().strip().split("\n")
      for word in words:
         for char in word.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
    char_counts = [(x,y) for x, y in char_counts.items()]
    itos = [x for x,y in sorted(char_counts, key=lambda z:(z[0],-z[1])) if y > 50]
    with open("/checkpoint/mhahn/char-vocab-wiki-"+args.language, "w") as outFile:
       print("\n".join(itos), file=outFile)
#itos = sorted(itos)
assert " " not in itos
itos.append(" ")
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

from config import CHECKPOINT_HOME

if args.load_from is not None:
  checkpoint = torch.load(CHECKPOINT_HOME+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      for chunk in data:
       print(len(chunk))
       for char in chunk:
#         if char == " ":
 #          continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
 #        if char not in stoi:
#             print(["OOV", char])
         numerified.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
       #  if len(numeric) > args.sequence_length:
        #    yield numeric
         #   numeric = [0]
       cutoff = int(len(numerified)/(args.batchSize*args.sequence_length)) * (args.batchSize*args.sequence_length)
       numerifiedCurrent = numerified[:cutoff]
       numerified = numerified[cutoff:]
       numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, args.sequence_length).transpose(0,1).transpose(1,2).cuda()
       #print(numerifiedCurrent.size())
       #quit()
       numberOfSequences = numerifiedCurrent.size()[0]
       for i in range(numberOfSequences):
#           print(numerifiedCurrent[i].size())
           yield numerifiedCurrent[i]
       hidden = None

def prepareDatasetChunksPrevious(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      for chunk in data:
       print(len(chunk))
       for char in chunk:
#         if char == " ":
 #          continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
 #        if char not in stoi:
#             print(["OOV", char])

         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric
            numeric = [0]






def prepareDataset(data, train=True):
      numeric = [0]
      count = 0
      for char in data:
#         if char == " ":
 #          continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
 #        if char not in stoi:
#             print(["OOV", char])

         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric
            numeric = [0]

hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

def forward(numeric, train=True, printHere=False):
      global hidden
      global beginning
      if hidden is None or (train and random.random() > 0.9):
          hidden = None
          beginning = zeroBeginning
      elif hidden is not None:
          hidden = tuple([Variable(x.data).detach() for x in hidden])

      numeric = torch.cat([beginning, numeric], dim=0)

      beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)
      

    #  print(char_embeddings)
      #if train and (embedding_full_dropout_prob is not None):
      #   embedded = embedded_dropout(char_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #char_embeddings(input_tensor)
      #else:
      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      out, hidden = rnn_drop(embedded, hidden)
#      if train:
#          out = dropout(out)

      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      
      loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

      if printHere and args.verbose:
         lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
         losses = lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()
#         boundaries_index = [0 for _ in numeric]
         print(("NONE", itos[numericCPU[0][0]-3]))
         for i in range(len(losses)):
 #           if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
  #             boundary = True
   #            boundaries_index[0] += 1
    #        else:
     #          boundary = False
            print((losses[i][0], itos[numericCPU[i+1][0]-3]))
      return [(losses[i][0], itos[numericCPU[i+1][0]-3]) for i in range(len(numericCPU)-1)]


import time

devLosses = []

import os

noiseType = "qa-corpus"
data_path = "../data/"+noiseType
texts = [x for x in os.listdir(data_path) if "text" in x and x.endswith(".txt")]

import codecs
rnn_drop.train(False)

with open("output/"+noiseType+"_"+args.load_from+".tsv", "w") as outFile:
 print("word"+"\t"+"surprisal"+"\t"+"nameText"+"\t"+"wordNum"+"\t"+"load_from"+"\t"+"noiseType", file=outFile)

 for text in texts:
   nameText = text[:text.index(".")]
   sents = codecs.open(data_path+"/"+text, 'r', 'iso-8859-1').read().strip()
   if "--\n--\n" in sents:
     sents = sents[:sents.index("--\n--\n")]
     sents = sents.split("\n")
     sents = " ".join(sents[5:]).lower()
   else:
     sents = sents[:sents.rfind("--\n")]
     sents = sents.split("\n")
     sents = " ".join(sents[5:]).lower()

   numerified = torch.cuda.LongTensor([(stoi[char]+3 if char in stoi else 2) for char in sents]).unsqueeze(1)
   print([char for char in sents if char not in stoi])


   surprisals = forward(numerified, printHere=True, train=False)
 #  print(len(surprisals), len(sents))
#   quit()
#   print(list(zip(surprisals, sents)))
   word = ""
   surprisal = 0
   wordNum = 0
   for out, char in zip(surprisals, sents):
      surprisal += out[0]
      if char not in ["'", '"']:
         word += char
      
      if char == " ":
#         print(word+" "+str(surprisal))
         print(word+"\t"+str(surprisal)+"\t"+nameText+"\t"+str(wordNum)+"\t"+args.load_from+"\t"+noiseType, file=outFile)

         word, surprisal = "", 0
         wordNum += 1

