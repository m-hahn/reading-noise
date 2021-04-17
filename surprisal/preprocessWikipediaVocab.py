path = "/private/home/mhahn/data/WIKIPEDIA/wikiextractor/enwiki/"

import os

vocab = {}

counter = 0
try:
  with open("/u/scr/mhahn/deepmindData/cnn-train.txt", "r") as inFile:
          for line in inFile:
             counter += 1
             if len(line) > 5 and not (line.startswith("<")):
               for char in line.lower():
                if char != " " and char != "\n":
                  vocab[char] = vocab.get(char, 0) + 1
             if counter % 10000 == 0:
                print("".join(sorted([x for x in vocab if vocab[x] > 10000])))
except KeyboardInterrupt:
   print(0)
with open("vocabularies/char-vocab-wiki-cnn.txt", "w") as outFile:
  itos = sorted([x[0] for x in sorted(list(vocab.items()), key=lambda x:x[1], reverse=True)[:70]])
  for char in itos:
      print(char, file=outFile)
#                print(line.strip(), file=outFile)


