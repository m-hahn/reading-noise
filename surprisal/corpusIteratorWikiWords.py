import random
 

def load(language, partition):
    chunk = []
    with open("/u/scr/mhahn/deepmindData/"+language+"-"+partition+".txt", "r") as inFile:
     for line in inFile:
      line = line.strip().lower().split(" ")
      chunk += line
      if len(chunk) > 40000:
      #   random.shuffle(chunk)
         yield chunk
         chunk = []
    yield chunk

def training(language):
  return load(language, "train")

def dev(language):
  return load(language, "dev")

def test(language):
  return load(language, "test")

