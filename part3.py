from collections import defaultdict
import sys
import json
import part2 as em
import numpy as np
from itertools import groupby
# defaultdict(int), if the key does not exist, it will initialize it to 0
# access something that doesnt exist, default value given is 0

def transition(file):

    tagCount = defaultdict(int)
    tagToWordDictionary = defaultdict(lambda: defaultdict(int))
    lines = file.readlines()

    # tagToWordDictionary structure: { y_i : { y_i-1 : count( y_i-1, y_i ) } }
    # tagCount: { u : count(u) }
    # mle: { y_i : y_i-1 : q(y_i, y_i-1) }


    # FIRST LINE
    first_line = lines[0]
    w = first_line.rstrip().rsplit(' ', 1)
    first_tag = w[1]
    tagToWordDictionary[first_tag]["START"] += 1
    tagCount[first_tag] += 1
    tagCount["START"] += 1

    for i in range(1, len(lines)):
        line = lines[i] # current line
        previous_line = lines[i-1] # previous line

        # STOP CASE : line = "", prev_line = "tag"
        if (line.rstrip() == ""):
            w_i1 = previous_line.rstrip().rsplit(' ', 1)

            tag_i_minus_1 = w_i1[1]
            tagToWordDictionary["STOP"][tag_i_minus_1] += 1
            tagCount["STOP"] += 1

        # START CASE : prev_line = "", line = "tag"
        if (previous_line.rstrip() == ""):
            w_i = line.rstrip().rsplit(' ', 1)

            tag_i = w_i[1]
            tagToWordDictionary[tag_i]["START"] += 1
            tagCount[tag_i] += 1
            tagCount["START"] += 1

        # case 3 = other words
        if (line.rstrip() != "") and (previous_line.rstrip() != ""):
            #   strip all trailing spaces first and then split by white spaces limit to 1
            w_i = line.rstrip().rsplit(' ', 1)
            w_i1 = previous_line.rstrip().rsplit(' ', 1)

            tag_i = w_i[1]
            tag_i_minus_1 = w_i1[1]

            tagToWordDictionary[tag_i][tag_i_minus_1] += 1
            tagCount[tag_i] += 1

    #calculate transition probs
    mle = defaultdict(int)
    for tag in tagCount:
        #   Create a dictionary for each tag which is y-values
        mle[tag] = {}
        #   list(tagToWordDictionary[tag]) returns us ALL the words (all x-values)
        for tag_i_minus_1 in list(tagToWordDictionary[tag]):
            mle[tag][tag_i_minus_1] = float(
                tagToWordDictionary[tag][tag_i_minus_1]) / (tagCount[tag_i_minus_1]) #count(u,v)/count(u)
    return mle

def inner_viterbi(sentence, em_params, tr_params, tags):
    words = sentence

    # FORWARD
    n = len(words)

    T= [[0 for i in range(len(tags))] for j in range(n) ]

    arg = [[0 for i in range(len(tags))] for j in range(n) ]


    # base case/ first layer
    for u in range(len(tags)):
        word = words[0].rstrip("\n")
        try:

            a = tr_params[tags[u]]["START"]
            b = em.getLikelihood(em_params, word, tags[u],w)

            if b == 0:
                nb = -sys.maxsize
            else:
                nb = np.log(b)

            T[0][u] = np.log(a)+nb

        except KeyError:

            if b == 0:
                nb = -sys.maxsize
            else:
                nb = np.log(b)
            T[0][u] = -sys.maxsize + nb

        except TypeError:
            print('base has TypeError')
            T[0][u] = -sys.maxsize


    for k in range(1, n):
        word = words[k].rstrip("\n")
        for v in range(len(tags)):
            find_max = []
            find_arg = []

            for u in range(len(tags)):
                try:
                    pilog = T[k-1][u]

                    a = tr_params[tags[v]][tags[u]]
                    b = em.getLikelihood(em_params, word, tags[v], w)

                    if b==0:
                        nb = -sys.maxsize
                    else:
                        nb = np.log(b)

                    value = pilog+np.log(a)+nb
                    arg_value = pilog+np.log(a)

                except KeyError:
                    if b == 0:
                        nb = -sys.maxsize
                    else:
                        nb = np.log(b)

                    value = pilog - sys.maxsize + nb
                    arg_value = pilog - sys.maxsize


                find_max.append(value)

                find_arg.append(arg_value)


            T[k][v] = max(find_max)
            arg[k][v] = find_arg.index(max(find_arg))+1



    #BACKWARD
    index = [0]*n
    y = [0]*n

    # Find maximum in last layer
    last_layer = T[n-1]
    final_check = []
    for i in range(len(last_layer)):
        pilog = last_layer[i]
        try:
            a = tr_params["STOP"][tags[i]]
            value = pilog + np.log(a)
        except KeyError:
            value = pilog - sys.maxsize
        final_check.append(value)

    # Find last tag
    true_max = -sys.maxsize
    for m in range(len(final_check)):
        value = final_check[m]
        if value > true_max:
            true_max = value
    if true_max == -sys.maxsize:
        index[n-1] = index_unknown
        y[n-1] = tags[index[n-1]]
    else:
        index[n-1] = final_check.index(true_max)+1
        y[n-1] = tags[final_check.index(true_max)]

    # Find the rest of values using stored arg table
    for j in range(n-1,0,-1):
        index[j-1] = arg[j][index[j]-1]
        y[j-1] = tags[index[j-1]-1]

    return y,T,arg

def viterbi(file, em_params, tr_params, tags):
    file_list = file.readlines()
    sentences = [list(group) for k, group in groupby(file_list, lambda x: x == "\n") if not k]
    total_y = []
    for i in range(len(sentences)):
        if i%10 == 0:
            print(i)
        sentence = sentences[i]
        y,T,arg = inner_viterbi(sentence, em_params, tr_params, tags)
        total_y += y
        total_y += [0]
    return total_y

def predictions_file(inputFile, outputfile, y):
    f = open(outputfile, 'w+', encoding="utf8")

    finput = open(inputFile, 'r', encoding="utf8")
    lines = finput.readlines()
    print(len(lines))
    print(len(y))
    for i in range(len(lines)):
        if (lines[i].rstrip() != ""):
            word = lines[i].rstrip()
            # print(word, y[i+1])
            if (word == '\n'):
                f.write(word)
            else:
                f.write(word+" "+y[i]+"\n")

        else:
            f.write('\n')
    f.close()
    print('Finished writing to File.')


if __name__ == "__main__":

    dataset = sys.argv[1]
    file_em = dataset+"/train"
    em_params, tagCount, w = em.emmissionWithSmoothing(open(file_em, "r", encoding="utf8"), 3)

    tr_params = transition(open(file_em, "r", encoding="utf8"))
    tags = list(tr_params)
    tags.remove("STOP")
    tags.remove("START")

    filePath = dataset+"/dev.in"
    fileout = dataset+"/dev.p3.out"

    common = ''
    max_count = 0
    for k,v in tagCount.items():
        if v > max_count:
            print(v)
            common = k
            max_count = v
    index_unknown = tags.index(common)

    y = viterbi((open(filePath, "r", encoding="utf8")), em_params, tr_params, tags)
    predictions_file(filePath, fileout, y)
