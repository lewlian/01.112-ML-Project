import sys
import part3 as tr
import part2 as em
import numpy as np
from itertools import groupby
import copy
from collections import Counter

def inner_viterbi_k(sentence, em_params, tr_params, tags, k):
    words = sentence

    # FORWARD
    n = len(words)

    T= [[[0] for i in range(len(tags))] for j in range(n) ]

    arg = [[[""] for i in range(len(tags))] for j in range(n) ]


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

            T[0][u] = [np.log(a)+nb]

        except KeyError:

            if b == 0:
                nb = -sys.maxsize
            else:
                nb = np.log(b)
            T[0][u] = [-sys.maxsize + nb]

        except TypeError:
            print('base has TypeError')
            T[0][u] = [-sys.maxsize]


    for j in range(1, n):
        word = words[j].rstrip("\n")

        for v in range(len(tags)):
            dic = {}

            for u in range(len(tags)):
                for t in range(len(T[j-1][0])):
                    try:
                        pilog = T[j-1][u][t]

                        a = tr_params[tags[v]][tags[u]]
                        b = em.getLikelihood(em_params, word, tags[v], w)

                        if b==0:
                            nb = -sys.maxsize
                        else:
                            nb = np.log(b)

                        value = pilog+np.log(a)+nb

                    except KeyError:
                        if b == 0:
                            nb = -sys.maxsize
                        else:
                            nb = np.log(b)

                        value = pilog - sys.maxsize + nb

                    path = arg[j-1][u][t]

                    dic[(u+1,path)] = value

            kt = Counter(dic)
            top_k = kt.most_common(k)
            top_k_prob = []
            top_k_path = []

            for i in range(k):
                top_k_prob.append(top_k[i][1])
                prev_path = top_k[i][0][1]
                new_path = prev_path + "," + str(top_k[i][0][0])
                top_k_path.append(new_path)

            T[j][v] = top_k_prob
            arg[j][v] = top_k_path



    #BACKWARD

    index = [0]*n
    y = [0]*n

    # Find k highest in last layer
    last_layer = T[n-1]
    final_check = {}
    for i in range(len(last_layer)):
        for t in range(k):
            pilog = last_layer[i][t]
            try:
                a = tr_params["STOP"][tags[i]]
                value = pilog + np.log(a)
            except KeyError:
                value = pilog - sys.maxsize

            final_check[(i,t)] = value
    final = Counter(final_check)
    top_final = final.most_common(k)

    # Find k best paths in stored arg table
    best_k_paths = []
    for m in range(k):
        i = top_final[m][0][0]
        t = top_final[m][0][1]
        prev_path = arg[n-1][i][t]
        total_path = prev_path+","+str(i+1)

        total_path_tags = []
        total_path_list = total_path.split(",")
        total_path_list = total_path_list[1:]

        for l in range(len(total_path_list)):
            total_path_tags.append(tags[int(total_path_list[l])-1])
        best_k_paths.append(total_path_tags)

    return best_k_paths,T,arg

def viterbi_kth(file, em_params, tr_params, tags, k):
    file_list = file.readlines()
    sentences = [list(group) for k, group in groupby(file_list, lambda x: x == "\n") if not k]
    total_y = []
    for i in range(len(sentences)):
        if i%10 == 0:
            print(i)
        sentence = sentences[i]
        y,T,arg = inner_viterbi_k(sentence, em_params, tr_params, tags, k)
        if len(y)!= k:
            print("Error: y from inner_viterbi_k is of length:",len(y),"while k is",k)
        total_y += y[k-1]
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
    k = int(sys.argv[2])
    file_em = dataset+"/train"
    em_params, tagCount, w = em.emmissionWithSmoothing(open(file_em, "r", encoding="utf8"), 3)

    tr_params = tr.transition(open(file_em, "r", encoding="utf8"))
    tags = list(tr_params)
    tags.remove("STOP")
    tags.remove("START")

    filePath = dataset+"/dev.in"
    fileout = dataset+"/dev.p4.out"

    y = viterbi_kth((open(filePath, "r", encoding="utf8")), em_params, tr_params, tags, k)
    predictions_file(filePath, fileout, y)
