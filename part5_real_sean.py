from collections import defaultdict
import sys
import os
import random


class perceptronTagger():
    def __init__(self, weight_counts):
        self.a = defaultdict(int)
        self.b = defaultdict(int)
        # self.classes = ["O", "I-positive", "B-positive",
        #                 "I-neutral", "B-neutral", "I-negative", "B-negative"]
        self.weight_counts = weight_counts

    def predict(self, word, prevTag):
        # scores = defaultdict(float)

        # Case for prevTag -> Stop
        if word == "" and prevTag != "":
            return ""
        # Case for Start -> word
        if word != "" and prevTag == "":
            if word in self.a:
                scores_for_word = {}
                # iterate through each tag of the word
                for tag in self.b[word]:
                    # multiply the emission with the
                    scores_for_word[tag] = (
                        self.b[word][tag] * self.a["START"][tag])
                return max(scores_for_word, key=scores_for_word.get)

        # Middle Scenario
        if word != "" and prevTag != "":
            if word in self.a:
                scores_for_word = {}
                # iterate through each tag of the word
                for tag in self.b[word]:
                    # multiply the emission with the
                    scores_for_word[tag] = (
                        self.b[word][tag] * self.a[prevTag][tag])
                return max(scores_for_word, key=scores_for_word.get)

    def train(self, n_iter, document):
        # document is basically the entire thing in a list[(word,tag),(word,tag)] for mat
        for i in range(n_iter):
            for i in range(1, len(document)):
                word = document[i][0]
                tag = document[i][1]
                prev_word = document[i-1][0]
                # NEED TO SCORE PREVIOUS GUESS, CANNOT USE THIS ONE
                prev_tag = document[i-1][1]

                if word == "":
                    guess = self.predict(word, prev_tag)
                    prev_tag2 = document[i-2][1]
                    guess2 = self.predict(prev_word, prev_tag2)
                    # CANNOT UPDATE REAL PREVIOUS TAG, MUST UPDATE PREDICTED PREVIOUS TAG
                    self.a[guess2]["STOP"] += 1

                if prev_word == "":
                    guess = self.predict(word, prev_tag)
                    self.a["START"][guess] += 1

                if prev_word != "" and word != "":
                    guess = self.predict(word, prev_tag)
                    # if the prediction is wrong
                    if guess != tag:
                       # To cover the first iteration when the dictionary a is still empty.
                        if word not in self.a:
                            self.b[word] = {"O": 0, "I-positive": 0, "B-positive": 0,
                                            "I-neutral": 0, "B-neutral": 0, "I-negative": 0, "B-negative": 0}
                        self.b[word][tag] += 1
                        self.b[word][guess] -= 1
<<<<<<< HEAD
                        self.a[prev_tag][guess] -=1
                        self.a[prev_tag][tag] += 1 
            # random.shuffle(document)
=======
                        self.a[prev_tag][tag]
            random.shuffle(document)
>>>>>>> 6c20a189f22e04e020b0d6dfff1f31ff2a8f0308


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


output, weight_counts = parse_feature_tag_pairs('./SG/', 'train')
print(output)
test = perceptronTagger(weight_counts)
Number of iterations to run perceptron
n = 10
test.train(n, output)
