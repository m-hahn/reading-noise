import random
 

def load(language, partition):
    chunks = []
    with open("/u/scr/mhahn/deepmindData/"+language+"-"+partition+".txt", "r") as inFile:
      for line in inFile:
        chunks.append(line.strip().lower())
        if len(chunks) > 20000:
#           random.shuffle(chunks)
           yield "".join(chunks)
           chunks = []
    yield "".join(chunks)

def training(language):
  return load(language, "train")

def dev(language):
  return load(language, "dev")

def test(language):
  return load(language, "test")

