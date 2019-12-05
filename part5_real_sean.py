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

    def predict(self, word):
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
                    scores_for_word[tag] = (b[word][tag] * a["START"][tag])
                return max(scores_for_word, key=scores_for_word.get)

        # Middle Scenario
        if word != "" and prevTag != "":
            if word in self.a:
                scores_for_word = {}
                # iterate through each tag of the word
                for tag in self.b[word]:
                    # multiply the emission with the
                    scores_for_word[tag] = (b[word][tag] * a[prevTag][tag])
                return max(scores_for_word, key=scores_for_word.get)
            #     for clas, weight in a.items():
            #         scores[clas] += weight
            # return max(self.classes, key=lambda clas: (scores[clas], clas))

    def train(self, n_iter, document):
        # document is basically the entire thin in a list[(word,tag),(word,tag)] for mat
        for i in range(n_iter):
            for word, tag in document:
                if word != "":
                    guess = self.predict(word)
                    # if the prediction is wrong
                    if guess != tag:
                       # Basically to cover the first it eration when the dictionary a is still empty.
                        if word not in self.a:
                            self.a[word] = {"O": 0, "I-positive": 0, "B-positive": 0,
                                            "I-neutral": 0, "B-neutral": 0, "I-negative": 0, "B-negative": 0}
                        self.a[word][tag] += 1
                        self.a[word][guess] -= 1
            random.shuffle(document)


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
test = perceptronTagger(weight_counts)
# Number of iterations to run perceptron
n = 10
test.train(n, output)
