from collections import defaultdict
import sys

#   emission without smoothing


def emmission(file):
    #   we use default dict so that we can do += operations on the value of dictionaries
    #   also with defaultdict(int), if the key does not exist, it will initialize it to 0
    tagCount = defaultdict(int)
    wordCount = defaultdict(int)
    tagToWordDictionary = defaultdict(lambda: defaultdict(int))

    for line in (file):

        if (line.rstrip() != ""):
            #   strip all trailing spaces first and then split by white spaces limit to 1
            w = line.rstrip().rsplit(' ', 1)
            currentWord = w[0]
            tag = w[1]

            #   dictionary of each tag to its words
            tagToWordDictionary[tag][currentWord] += 1
            #   dictionary of the count of each unique word
            wordCount[currentWord] += 1
            #   dictionary of the count of each unique tag (O, B-positive, I-positive, etc.)
            tagCount[tag] += 1

    e = {}
    for tag in tagCount:
        #   Create a dictionary for each tag which is y-values
        e[tag] = {}
        #   list(tagToWordDictionary[tag]) returns us ALL the words (all x-values)
        for currentWord in list(tagToWordDictionary[tag]):
            e[tag][currentWord] = float(
                tagToWordDictionary[tag][currentWord]) / (tagCount[tag])
    return e


#   emission with smoothing function
def emmissionWithSmoothing(file, k):
    # we use default dict so that we can do += operations on the value of dictionaries
    tagCount = defaultdict(int)
    wordCount = defaultdict(int)
    tagToWordDictionary = defaultdict(lambda: defaultdict(int))

    for line in (file):

        if (line.rstrip() != ""):
            #   strip all trailing spaces first and then split by white spaces limit to 1
            w = line.rstrip().rsplit(' ', 1)
            currentWord = w[0]
            tag = w[1]
            #   dictionary of each tag to its words
            tagToWordDictionary[tag][currentWord] += 1
            #   dictionary of the count of each unique word
            wordCount[currentWord] += 1
            #   dictionary of the count of each unique tag (O, B-positive, I-positive, etc.)
            tagCount[tag] += 1

    for word in list(wordCount):
        #   if the total word count of a word is less than k, we replace it with #UNK# with the same count
        if wordCount[word] < k:
            #print(word + '\n')
            wordCount["#UNK#"] += wordCount[word]
            del wordCount[word]
            #   search through the y->x table and if there is that word, we replace it with #UNK#
            for tag in list(tagToWordDictionary):
                if word in tagToWordDictionary[tag]:
                    tagToWordDictionary[tag]['#UNK#'] += tagToWordDictionary[tag][word]
                    del tagToWordDictionary[tag][word]

    e = defaultdict(lambda: defaultdict(float))

    for tag in tagCount:
        # list(tagToWordDictionary[tag]) returns us ALL the words (all x-values)
        for word in list(tagToWordDictionary[tag]):
            e[tag][word] = float(
                tagToWordDictionary[tag][word]) / (tagCount[tag])
    # for tag in tagCount:
    #     print(tag, e[tag]["#UNK#"])
    print(tagCount)
    return e, tagCount, wordCount


def predictLabel(e, tags, w, inputFile, outputFile):
    f = open(outputFile, 'w+', encoding="utf8")
    for line in open(inputFile, 'r', encoding="utf8"):
        if (line.rstrip() != ""):
            word = line.rstrip()
            eForWord = []
            for tag in tags:
                eForWord.append((getLikelihood(e, word, tag, w), tag))
            f.write(word+" "+max(eForWord)[1]+"\n")
            if max(eForWord)[0] == 0:
#                 print(word, max(eForWord)[0])
                print('ERROR')
        else:
            f.write('\n')
    f.close()
    print('Finished writing to File.')


def getLikelihood(e, word, tag, w):
    # check if word existed in training, ONLY if it doesn't then we return #UNK#, else if it was trained but not in the tag then 0
    key_list = list(w.keys())
#     print(key_list)
#     print(word)
#     print(word in key_list)
    if word in w:
        #         print('word in keys')
        if word in e[tag]:
            #             print('word in tag')
            out = e[tag][word]
#             print('GLI:',out)
        else:
            #             print('trained but not in tag')

            out = 0
#             print('GLI:',out)
    else:
        #         print('unknown:',word)
        out = e[tag]["#UNK#"]
#         print('GLI:',out)
#     print('GLI_final:',out)
    return out


if __name__ == "__main__":
    filePath = sys.argv[1]
    e, t, w = emmissionWithSmoothing(
        open(filePath+"/train", "r", encoding="utf8"), 3)
    predictLabel(e, t, w, filePath+"/dev.in", filePath+"/dev.p2.out")
