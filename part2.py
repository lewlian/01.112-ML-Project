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

    mle = {}
    for tag in tagCount:
        #   Create a dictionary for each tag which is y-values
        mle[tag] = {}
        #   list(tagToWordDictionary[tag]) returns us ALL the words (all x-values)
        for currentWord in list(tagToWordDictionary[tag]):
            mle[tag][currentWord] = float(
                tagToWordDictionary[tag][currentWord]) / (tagCount[tag])
    return mle


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

    mle = defaultdict(lambda: defaultdict(float))

    for tag in tagCount:
        # list(tagToWordDictionary[tag]) returns us ALL the words (all x-values)
        # print(tagToWordDictionary[tag]["#UNK#"])
        for word in list(tagToWordDictionary[tag]):
            mle[tag][word] = float(
                tagToWordDictionary[tag][word]) / (tagCount[tag])
    # for tag in tagCount:
    #     print(tag, mle[tag]["#UNK#"])
    return mle, tagCount


def predictLabel(mle, tags, inputFile, outputFile):
    f = open(outputFile, 'w+', encoding="utf8")
    for line in open(inputFile, 'r', encoding="utf8"):
        if (line.rstrip() != ""):
            word = line.rstrip()
            mleForWord = []
            for tag in tags:
                mleForWord.append((getLikelihood(mle, word, tag), tag))
            f.write(word+" "+max(mleForWord)[1]+"\n")
            #print(word, max(mleForWord)[1])
        else:
            f.write('\n')
    f.close()
    print('Finished writing to File.')


def getLikelihood(mle, word, tag):
    if word in mle[tag]:
        return mle[tag][word]
    else:
        return mle[tag]["#UNK#"]


if __name__ == "__main__":
    filePath = sys.argv[1]
    mle, t = emmissionWithSmoothing(
        open(filePath+"/train", "r", encoding="utf8"), 3)
    predictLabel(mle, t, filePath+"/dev.in", filePath+"/dev.prediction.out")
