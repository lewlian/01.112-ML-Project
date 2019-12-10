from collections import defaultdict
import sys
import os
import random
import string


class perceptronTagger():
    # Initialises a perceptronTagger instance with required dictionaries
    def __init__(self, tag_counts):
        self.tag_counts = tag_counts
        self.classes = tag_counts.keys()  # get keys
        self.weights = defaultdict(lambda: defaultdict(int))

    # features here represent a word, we created a list of features to represent each word
    def predict(self, features):
        if features != "":
            # total score keep track of the score for each tag related to the word
            scores = defaultdict(int)
            for feature in features:
                # if the particular feature is not found in our model's weights class, we ignore
                if feature not in self.weights:
                    continue
                # else if it is found, select the dictionary for the feature from the weights class
                weights = self.weights[feature]
                # we ultimately want to use the tag with the highest score as our prediction, thus we increment it and keep track in a dictionary called scores
                for tag, weight in weights.items():
                    scores[tag] += weight
            # we return the tag with the highest score as our prediction
            return max(self.classes, key=lambda tag: (scores[tag], tag))
        return ""

    def train(self, iter, document):
        for i in range(iter):
            print("Training for iteration...", i)


# pass in the features and correct tag to predict function
            for features, correct_tag in document:
                predicted_tag = self.predict(features)
                # update weights only if there is any wrong prediction
                if predicted_tag != correct_tag:
                    for feature in features:
                        self.weights[feature][predicted_tag] -= 1
                        self.weights[feature][correct_tag] += 1
            random.shuffle(document)

        for feature in self.weights:
            for tag in self.weights[feature]:
                self.weights[feature][tag] = self.weights[feature][tag] / \
                    (iter*len(document))

        return self.weights


def parse_predict_test_file(fileIn, fileOut, model):
    fout = open(fileOut, 'w+', encoding="utf8")
    # list_new_guesses = [""]
    finput = open(fileIn, 'r', encoding="utf8")
    lines = finput.readlines()

    output = []
    for i in range(0, len(lines)):
        # special case for first line: previous and previous previous word features have to be specificially handled since they are null
        if i == 0:
            line = lines[i].strip().split(" ")
            next_line = lines[i+1].strip().split(" ")
            next2_line = lines[i+2].strip().split(" ")
            features = get_features(
                line[0], "", "", "", "", next_line[0], next2_line[0])
            guess = model.predict(features)
            output.append(guess)
            fout.write(line[0]+" "+guess+"\n")
        # special case for second line: previous word features have to be specifically handled since they are null
        elif i == 1:
            line = lines[i].strip().split(" ")
            next_line = lines[i+1].strip().split(" ")
            next2_line = lines[i+2].strip().split(" ")
            prev_line = lines[i-1].strip().split(" ")
            prev_word = prev_line[0]
            prev_tag = output[0]
            features = get_features(
                line[0], prev_word, prev_tag, "", "", next_line[0], next2_line[0])
            guess = model.predict(features)
            output.append(guess)
            fout.write(line[0]+" "+guess+"\n")
        # case for the rest of the document (middle case)
        else:
            if lines[i].strip() != "":
                # read in current current line
                line = lines[i].strip().split(" ")
                # read in previous 2 lines
                prev_line = lines[i-1].strip().split(" ")
                prev2_line = lines[i-2].strip().split(" ")

                # extract features related to current line and previous 2 lines
                word = line[0]
                prev_word = prev_line[0]
                prev_tag = output[i-1]
                prev2_tag = output[i-2]
                prev2_word = prev2_line[0]

                # reading and extracting for next 2 line features
                # special case if we are at the last line of the document:  next word and next next word will be null
                if i == len(lines)-1:
                    next_word = ""
                    next2_word = ""
                # special case if we are at the second last line of the document: next next word will be null
                elif i == len(lines)-2:
                    next_line = lines[i+1].strip().split(" ")
                    next_word = next_line[0]
                    next2_word = ""
                # all other cases in the  middle of the document
                else:
                    next_line = lines[i+1].strip().split(" ")
                    next2_line = lines[i+2].strip().split(" ")
                    next_word = next_line[0]
                    next2_word = next2_line[0]

                # pass in the features to get_features to return a set of feature that we can pass in to predict function
                features = get_features(
                    word, prev_word, prev_tag, prev2_tag, prev2_word, next_word, next2_word)
                guess = model.predict(features)
                output.append(guess)
                fout.write(line[0]+" "+guess+"\n")

            else:
                output.append("")
                fout.write("\n")

    return


def parse_feature_tag_pairs(folder_path, filename):
    output = []
    output.append(("", ""))

    tag_counts = defaultdict(int)
    finput = open(os.path.join(folder_path, filename), 'r', encoding="utf8")
    lines = finput.readlines()

    # handling of special cases is the same as "parse_predict_test_file"
    for i in range(0, len(lines)):
        if i == 0:
            line = lines[i].strip().split(" ")
            next_line = lines[i+1].strip().split(" ")
            next2_line = lines[i+2].strip().split(" ")
            features = get_features(
                line[0], "", "", "", "", next_line[0], next2_line[0])
            output.append((features, line[1]))
        elif i == 1:
            line = lines[i].strip().split(" ")
            next_line = lines[i+1].strip().split(" ")
            prev_line = lines[i-1].strip().split(" ")
            next2_line = lines[i+2].strip().split(" ")
            prev_word = prev_line[0]
            prev_tag = prev_line[1]
            features = get_features(
                line[0], prev_word, prev_tag, "", "", next_line[0], next2_line[0])
            output.append((features, line[1]))

        else:
            if lines[i].strip() != "":
                line = lines[i].strip().split(" ")
                if i == len(lines)-1:
                    next_word = ""
                    next2_word = ""
                elif i == len(lines)-2:
                    next_line = lines[i+1].strip().split(" ")
                    next_word = next_line[0]
                    next2_word = ""
                else:
                    next_line = lines[i+1].strip().split(" ")
                    next_word = next_line[0]
                    next2_line = lines[i+2].strip().split(" ")
                    next2_word = next2_line[0]

                prev_line = lines[i-1].strip().split(" ")
                prev2_line = lines[i-2].strip().split(" ")
                word = line[0]
                tag = line[1]
                prev_word = prev_line[0]
                prev2_word = prev2_line[0]

                if(prev_word == ''):
                    prev_tag = ""
                else:
                    prev_tag = prev_line[1]

                if(prev2_word == ""):
                    prev2_tag = ""
                else:
                    prev2_tag = prev2_line[1]

                features = get_features(
                    word, prev_word, prev_tag, prev2_tag, prev2_word, next_word, next2_word)

                output.append((features, tag))
                tag_counts[tag] += 1
            else:
                output.append(("", ""))

    return output, tag_counts


def isFirstCapital(token):  # check if first letter is capital
    if token[0].upper() == token[0]:
        return "yes"
    else:
        return "no"


def isAlpha(word):  # check if the word contains digits or punctuation
    if word.isalpha():
        return "yes"
    else:
        return "no"


def get_features(word, prev_word, prev_tag, prev2_tag, prev2_word, next_word, next2_word):
    def add(name, *args):
        features.add('+'.join((name,) + tuple(args)))  # generate the features
    # set can only have unique elements.
    features = set()
    add("isFirstCapital", isFirstCapital(word))
    add("isAlpha", isAlpha(word))

    # convert to lower case for better performance
    word = word.lower()
    prev_word = prev_word.lower()
    prev2_word = prev2_word.lower()
    next_word = next_word.lower()
    next2_word = next2_word.lower()

    add('iSuffix', word[-3:])  # suffix of current word
    add('iSuffix1', word[-2:])  # suffix of current word
    add("iPrefix", word[0:2])  # prefix of current word
    add("i-1Tag", prev_tag)
    add("iWord", word)
    add("i-1Word", prev_word)
    add("i-2Word", prev2_word)
    add("i-2Tag", prev2_tag)
    add("i+1Word", next_word)
    add("i+2Word", next2_word)
    add("i-1Suffix", prev_word[-3:])  # suffix of previous word
    add("i+1Suffix", next_word[-3:])  # suffix of next word
    return features


# RUNNING THE CODE
output, tag_counts = parse_feature_tag_pairs('./AL/', 'train')
test = perceptronTagger(tag_counts)

# Number of iterations to run perceptron
n = 20
model_weights = test.train(n, output)
fileIn = './EN/dev.in'
fileOut = './EN/devavg.p5.out'
parse_predict_test_file(fileIn, fileOut, test)


# 20 iter, no avg : 0.8951, 0.9007 0.8948
# 20 iter, avg : 0.9035, 0.8962
