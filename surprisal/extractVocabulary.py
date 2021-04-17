unigrams = {}

language = "dailymail"
partition = "train"
import random
counter = 0
with open("/u/scr/mhahn/deepmindData/"+language+"-"+partition+".txt", "r") as inFile:
   for line in inFile:
      counter += 1
      if counter % 10000 == 0:
         print(counter)
      line = line.strip().lower().split(" ")
      for word in line:
         if len(word) == 0:
           continue
         unigrams[word] = unigrams.get(word, 0) + 1
 #     if random.random() > 0.99:
#          break
unigrams = sorted(list(unigrams.items()), key=lambda x:x[1],reverse=True)
with open("vocabularies/dailymail-word-vocab-50000.txt", "w") as outFile:
  for word, count in unigrams[:50000]:
      print(f"{word}\t{count}", file=outFile)
      

