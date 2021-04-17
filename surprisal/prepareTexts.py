import os, sys

part = sys.argv[1]

pathOut = "/u/scr/mhahn/deepmindData/"+part
path = "/u/scr/mhahn/deepmindData/"+part+"/stories/"
files = os.listdir(path)

import random
random.shuffle(files)
i = 0
with open(pathOut+"-train.txt", "w") as train:
 with open(pathOut+"-dev.txt", "w") as dev:
  with open(pathOut+"-test.txt", "w") as test:
   for name in files:
      i += 1
      if i % 1000:
         print(float(i)/len(files))
      with open(path+name, "r") as inFile:
        r = random.random()
        if r < 0.02:
          partition = dev
        elif r < 0.04:
          partition = test
        else:
          partition = train
        for line in inFile:
          if line.startswith("@highlight"):
             break
          if len(line) < 3:
            continue
          print >> partition, line.strip()
