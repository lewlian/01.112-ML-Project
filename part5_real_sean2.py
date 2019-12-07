from collections import defaultdict
import sys
import os
import random


class perceptronTagger():
    def __init__(self, weight_counts):
        self.a = defaultdict(lambda: defaultdict(int))
        self.b = defaultdict(lambda: defaultdict(int))
        self.weight_counts = weight_counts
        self.classes = weight_counts.keys()  # get keys

    def predict(self, word, prevTag):
        # scores = defaultdict(float)
        # print(word, prevTag)
        # Case for prevTag -> Stop
        if word == "" and prevTag != "":
            return ""

        # Case for Start -> word
        if word != "" and prevTag == "":
            scores_for_word = {}
            # iterate through each tag of the word
            if word in self.b:
                for tag in self.classes:
                    # multiply the emission with the
                    scores_for_word[tag] = (
                        self.b[word][tag] * self.a["START"][tag])
            else:
                scores_for_word["O"] = 0
            # print(scores_for_word)
            return max(scores_for_word, key=scores_for_word.get)

        # Middle Scenario
        if word != "" and prevTag != "":
            scores_for_word = {}
            # iterate through each tag of the word
            if word in self.b:
                for tag in self.classes:
                    # multiply the emission with the
                    scores_for_word[tag] = (
                        self.b[word][tag] * self.a[prevTag][tag])
            else:
                scores_for_word["O"] = 0
            # print(scores_for_word)
            return max(scores_for_word, key=scores_for_word.get)

    def train(self, n_iter, document):
        # document is basically the entire thing in a list[(word,tag),(word,tag)] for mat
        for i in range(n_iter):

            # Stores a list of the guesses so that we can use it in predict (and not use the actual one)
            # initialize with a "" because the TAG of the very first word should be START
            list_of_guesses = [""]

            for i in range(1, len(document)):
                word = document[i][0]
                tag = document[i][1]
                prev_word = document[i-1][0]
                # NEED TO SCORE PREVIOUS GUESS, CANNOT USE THIS ONE
                prev_tag = document[i-1][1]

                guess = self.predict(word, list_of_guesses[i-1])
                list_of_guesses.append(guess)
                if prev_word == "":
                    self.a["START"][guess] += 1

                    if guess != tag:
                       # To cover the first iteration when the dictionary a is still empty.
                        # if word not in self.a:
                        #     self.b[word] = {"O": 0, "I-positive": 0, "B-positive": 0,
                        #                     "I-neutral": 0, "B-neutral": 0, "I-negative": 0, "B-negative": 0}
                        self.b[word][tag] += 1
                        self.b[word][guess] -= 1
                        self.a[list_of_guesses[i-1]][guess] -= 1
                        self.a[prev_tag][tag] += 1

                if prev_word != "" and word != "":

                    # if the prediction is wrong
                    if guess != tag:
                       # To cover the first iteration when the dictionary a is still empty.
                        # if word not in self.a:
                        #     self.b[word] = {"O": 0, "I-positive": 0, "B-positive": 0,
                        #                     "I-neutral": 0, "B-neutral": 0, "I-negative": 0, "B-negative": 0}
                        self.b[word][tag] += 1
                        self.b[word][guess] -= 1
                        self.a[list_of_guesses[i-1]][guess] -= 1
                        self.a[prev_tag][tag] += 1

        self.b["#UNK#"] = defaultdict(int)
        for word in self.b.keys():
            total_count_for_word = 0
            for tag in self.b[word]:
                total_count_for_word += self.b[word][tag]

            if total_count_for_word <= 3:
                for tag in self.b[word]:
                    self.b['#UNK#'][tag] += self.b[word][tag]

        return list_of_guesses, self.a, self.b


def predict_test(fileIn, a, b, classes, fileOut):
    print(classes)
    f = open(fileOut, 'w+', encoding="utf8")

    list_new_guesses = [""]

    finput = open(fileIn, 'r', encoding="utf8")
    lines = finput.readlines()

    for i in range(len(lines)):
        word = lines[i].rstrip()
        # tag for word i is list_new_guesses[i+1]
        guess = predict_tag_test(word, list_new_guesses[i], a, b, classes)
        list_new_guesses.append(guess)
        f.write(word+" "+guess+"\n")

    f.close()


def predict_tag_test(word, prevTag, a, b, classes):
        # scores = defaultdict(float)
        # print(word, prevTag)
        # Case for prevTag -> Stop
    if word == "":
        return "\n"

    # Case for Start -> word
    if word != "" and prevTag == "":
        scores_for_word = {}
        # if we train on the word before
        if word in b:
            # iterate through every tag of the word, it is a default dict so even if it doesn't exist it will return 0
            for tag in classes:
                    # multiply the emission with the transition value
                scores_for_word[tag] = (
                    b[word][tag] * a["START"][tag])

        # if we did not train on the word before
        else:
            for tag in classes:
                    # multiply the emission with the transition value for #UNK#
                scores_for_word[tag] = (
                    b["#UNK#"][tag] * a["START"][tag])

        # return the tag with the max score for that word
        return max(scores_for_word, key=scores_for_word.get)

    # Middle Scenario
    if word != "" and prevTag != "":
        scores_for_word = {}
        # iterate through each tag of the word
        if word in b:
            for tag in classes:
                # multiply the emission with the
                scores_for_word[tag] = (
                    b[word][tag] * a[prevTag][tag])

        # if we did not train on the word before
        else:
            for tag in classes:
                    # multiply the emission with the transition value for #UNK#
                scores_for_word[tag] = (
                    b["#UNK#"][tag] * a["START"][tag])

        # return the tag with the max score for that word
        return max(scores_for_word, key=scores_for_word.get)


def parse_feature_tag_pairs(folder_path, filename):
    output = []
    tag_counts = defaultdict(int)
    with open(os.path.join(folder_path, filename), 'r', encoding="utf8") as infile:
        # For the very first start of the document

        output.append(("", ""))
        for line in infile:
            if line.strip() != "":
                proc_line = line.strip().split(" ")
                word = proc_line[0]
                tag = proc_line[1]
                output.append((word, tag))
                tag_counts[tag] += 1
            else:
                output.append(("", ""))

    return output, tag_counts

# def predictions_file(inputFile, outputfile, y):
#     f = open(outputfile, 'w+', encoding="utf8")

#     finput = open(inputFile, 'r', encoding="utf8")
#     lines = finput.readlines()

#     for i in range(len(lines)):
#         if (lines[i].rstrip() != ""):
#             word = lines[i].rstrip()
#             # print(word, y[i+1])
#             f.write(word+" "+y[i+1]+"\n")

#         else:
#             f.write('\n')
#     f.close()
#     print('Finished writing to File.')


output, weight_counts = parse_feature_tag_pairs('./EN/', 'train')
# print(output)
test = perceptronTagger(weight_counts)
# Number of iterations to run perceptron
n = 10
guesses, a_mp, b_mp = test.train(n, output)

fileIn = './EN/dev.in'
fileOut = './EN/dev.p5.out'

predict_test(fileIn, a_mp, b_mp, test.classes, fileOut)


# predictions_file(filePath, fileout, guesses)
