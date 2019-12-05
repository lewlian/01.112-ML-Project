from collections import defaultdict
import random
import os

# set n here
n = 10


class perceptronTagger():
    def __init__(self, weight_counts):
        self.weights = {}
        self.classes = ["O", "I-positive", "B-positive",
                        "I-neutral", "B-neutral", "I-negative", "B-negative"]
        self.weight_counts = weight_counts

    def predict(self, word):
        scores = defaultdict(float)
        if word != "":
            if word in self.weights:
                weights = self.weights[word]
                for clas, weight in weights.items():
                    scores[clas] += weight
            return max(self.classes, key=lambda clas: (scores[clas], clas))
        return ""

    def train(self, n_iter, examples):
        for i in range(n_iter):
            for word, tag in examples:
                if word != "":
                    guess = self.predict(word)
                    # if the prediction is wrong
                    if guess != tag:
                       # Basically to cover the first it eration when the dictionary weights is still empty.
                        if word not in self.weights:
                            self.weights[word] = {"O": 0, "I-positive": 0, "B-positive": 0,
                                                      "I-neutral": 0, "B-neutral": 0, "I-negative": 0, "B-negative": 0}
                        self.weights[word][tag] += 1
                        self.weights[word][guess] -= 1
            random.shuffle(examples)


def parse_features(folder_path, filename):
    output = []
    with open(os.path.join(folder_path, filename), 'r', encoding="utf8") as infile:
        for line in infile:
            proc_line = line.strip()
            if proc_line != "":
                output.append(proc_line)
            else:
                output.append("")
    return output


def parse_feature_tag_pairs(folder_path, filename):
    output = []
    weight_counts = {"O": 0, "I-positive": 0, "B-positive": 0,
                     "I-neutral": 0, "B-neutral": 0, "I-negative": 0, "B-negative": 0}
    with open(os.path.join(folder_path, filename), 'r', encoding="utf8") as infile:
        for line in infile:
            if line.strip() != "":
                proc_line = line.strip().split(" ")
                output.append((proc_line[0], proc_line[1]))
                weight_counts[proc_line[1]] += 1
            else:
                output.append(("", ""))
    return output, weight_counts


SG_folder = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data/SG")
CN_folder = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data/CN")
ES_folder = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data/ES")
EN_folder = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data/EN")
ESt_folder = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data\ES-test")
ENt_folder = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data\EN-test")

folders_to_run = [SG_folder, CN_folder, ES_folder, EN_folder]
test_folders_to_run = [ESt_folder, ENt_folder]


for folder in folders_to_run:

    feature_pairs, weight_counts = parse_feature_tag_pairs(folder, "train")

    test = perceptronTagger(weight_counts)
    test.train(n, feature_pairs)

    with open(os.path.join(folder, "test.p5.out"), 'w', encoding="utf8") as outfile:
        for word in parse_features(folder, "dev.in"):
            outfile.write(word + " " + test.predict(word)+"\n")


feature_pairs, weight_counts = parse_feature_tag_pairs(ES_folder, "train")

test = perceptronTagger(weight_counts)
test.train(10, feature_pairs)

with open(os.path.join(ESt_folder, "test.p5.out"), 'w', encoding="utf8") as outfile:
    for word in parse_features(ESt_folder, "test.in"):
        outfile.write(word + " " + test.predict(word) + "\n")

feature_pairs, weight_counts = parse_feature_tag_pairs(EN_folder, "train")

test = perceptronTagger(weight_counts)
test.train(n, feature_pairs)

with open(os.path.join(ENt_folder, "test.p5.out"), 'w', encoding="utf8") as outfile:
    for word in parse_features(ENt_folder, "test.in"):
        outfile.write(word + " " + test.predict(word) + "\n")
