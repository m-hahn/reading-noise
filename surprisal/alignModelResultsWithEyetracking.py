#model results

modelOutput = [x.split("\t") for x in open("/juicier/scr120/scr/mhahn/CODE/char-lm-noise/surprisal/output/qa-corpus_test-661158218.tsv", "r").read().strip().split("\n")]

eyetrackingOutput = [x.split("\t") for x in open("/juicier/scr120/scr/mhahn/CODE/char-lm-noise/eyetracking/results/exp1_list1.fp.Rdata.utf8", "r").read().strip().split("\n")]


textNumbersConversion = dict([tuple(x.split("\t")[::-1]) for x in open("textNumbersConversion.tsv", "r").read().strip().split("\n")[1:]])

print(modelOutput[1])
print(eyetrackingOutput[0])
#quit()

with open("modelEyetrackingTokenAlignment.tsv", "w") as outFile:
 print("\t".join(["WordEyetracking", "WordModel", "TextEyetracking", "TextModel", "IndexEyetracking", "IndexModel"]), file=outFile)
 for text, number in textNumbersConversion.items():
#   if number == "110":
 #     continue
   print(text, number)
   text = text.replace(".txt", "")
   fromEyetracking = [x for x in eyetrackingOutput if str(int(int(x[1])/10)) == number]
   fromModel = [x for x in modelOutput if x[2] == text]
   print(len(fromEyetracking)/len(fromModel))
   indexEye = 0
   indexModel = 0
   failures = 0
   if number == "110": # a failure of the script reading texts for the neural networks led to the first line of Dailymail 153.txt being ignored there.
       indexEye = 12
   while indexEye < len(fromEyetracking) and indexModel < len(fromModel):
       wordEye = fromEyetracking[indexEye][14]
       wordModel = fromModel[indexModel][0]
       if len(wordModel.lower().strip()) == 0:
          indexModel += 1
          continue
       if len(wordEye.lower().strip()) == 0:
          indexEye += 1
          continue
       indexFromEyetracking = int(fromEyetracking[indexEye][2])
       indexFromModel = int(fromModel[indexModel][3])
       print("\t".join([str(x) for x in [wordEye, wordModel, number, text, indexFromEyetracking, indexFromModel]]), file=outFile)

       if wordEye.lower().strip() != wordModel.lower().strip():
          failures += 1
       else:
          failures = 0
       assert failures < 10
       #assert wordEye == wordModel, (wordEye, wordModel)
       indexEye += 1
       indexModel += 1


