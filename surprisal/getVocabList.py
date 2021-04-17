language = "dailymail"
partition = "train"

import re

# https://stackoverflow.com/questions/8687018/python-string-replace-two-things-at-once
def replace_all(repls, str):
    # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], str)                                     
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], str)                                     

vocab = {}
text =  "i like apples, but pears scare me"
print replace_all({"apple": "pear", "pear": "apple"}, text)
replacements = {". " : " . ", ", " : " , ", "; " : " ; ", ": " : " : ", "! " : " ! ", "? " : " ? ", ") " : " ) ", " (" : " ( "}
count = 0
with open("/u/scr/mhahn/deepmindData/"+language+"-"+partition+".txt", "r") as inFile:
   for line in inFile:
     line = replace_all(replacements, (line+" ").lower()).split(" ")
     for word in line:
       if len(word) == 0:
         continue
       vocab[word] = vocab.get(word, 0) + 1
     count += 1
     if count % 10000 == 0:
         print(count)
#         break

words = sorted(list(vocab.iteritems()), key=lambda x:x[1], reverse=True)
with open("/u/scr/mhahn/dailyail-vocab.txt", "w") as outFile:
   for word, count in words: 
      print >> outFile, word+"\t"+str(count)
