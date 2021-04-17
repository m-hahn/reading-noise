import os
import sys


path = "/juicier/scr120/scr/mhahn/CODE/char-lm-noise/noise_corpora/qa-corpus"
files = os.listdir(path)
textsGround = []
for name in files:
   with open(path+"/"+name, "r") as inFile:
      textsGround.append((name, set(inFile.read().strip().replace("\n", "").split(" "))))

path = "/juicier/scr120/scr/mhahn/CODE/char-lm-noise/eyetracking/results/"
files = [x for x in os.listdir(path) if x.endswith(".fp.Rdata.utf8")]
texts = {}
for name in files:
   with open(path+name, "r", encoding="utf-8") as inFile:
      for line in inFile:
          line = line.strip().split("\t")
          textNum = int(int(line[1])/10)
          word = line[14]
          if textNum not in texts:
             texts[textNum] = set()
          texts[textNum].add(word)
with open("textNumbersConversion.tsv", "w") as outFile:
  covered = set()
  print("\t".join(["ExperimentTextNumber", "FileName"]), file=outFile)
  for textNumber, text in texts.items():
     candidates = [(name, (len(text.intersection(text0))/len(text.union(text0))) , len(text)/len(text0)) for name, text0 in textsGround]
     best = max([x[1] for x in candidates])
     bestText = [i for i in range(len(candidates)) if candidates[i][1] == best][0]
     print(textNumber, best, bestText)
     assert bestText not in covered
     covered.add(bestText)
     print("\t".join([str(textNumber), candidates[bestText][0]]), file=outFile)
#print(texts)

