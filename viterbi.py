import part3 as tr
import part2 as em
import numpy as np
import json
# model parameters

# em_params, tags = em.emmissionWithSmoothing(file, k)
# b_v(x) = em_params[v][x] (v=tag, x=word)

# tr_params = tr.transition(file)
# a(u, v) = tr_params[v][u]



def viterbi(file, em_params, tr_params, tags, w):
    words = file.readlines()

    # FORWARD PROPAGATION
    n = len(words)
    print(n)
    T = [[0 for i in range(len(tags))] for i in range(n) ]
    # print(tr_params["STOP"])
    # base case
    for u in range(len(tags)):
        # print(words[0])
        word = words[0].strip("\n")
        emission = em.getLikelihood(em_params, word, tags[u], w)
        # print(emission)
        if(emission!=0):
            try:
                T[0][u] = np.log(tr_params[tags[u]]["START"]) + np.log(em.getLikelihood(em_params, word[0], tags[u], w))
                # T[0][u] = 0
            except KeyError:
                T[0][u] = 0
            except TypeError:
                T[0][u] = 0
        else:
            T[0][u] = 0

    # f = open("T.txt", "w")
    
    for k in range(1, n):
        word = words[k].strip("\n")

        for v in range(len(tags)):
            find_max = []
            # print("word ", word)
            b = em.getLikelihood(em_params, word, tags[v], w)
            # print("b ",b)
            if(b==0):
                T[k][v] = 0
            else:
                b = np.log(b)
                for u in range(len(tags)):
                    pi = T[k-1][u]
                    try:
                        a = np.log(tr_params[tags[v]][tags[u]])
                        value = pi+a+b
                        find_max.append(value)
                    except KeyError:
                        find_max.append(-10000000)

                T[k][v] = max(find_max)
        
    
    # print("Finished forward propagation")

    # BACKWARD PROPAGATION

    # find last layer with no zeros
    z = n-1
    y = [0 for x in range(n)]
    max_tag = ""
    
    max_value = -1000000

    for u in range(len(tags)):
        if(T[n-1][u]==0):
            x = -1000000
        else:
            try:
                x = (T[n-1][u]+np.log(tr_params["STOP"][tags[u]]))
            except KeyError:
                x = -1000000
        
        # print(tags[u], x)
        if (x>max_value):
            max_value = x
            max_tag = tags[u]
    
    y[n-1] = max_tag
    # print(y)


    
    for m in range(n-2, -1, -1):
        max_y = -1000000
        max_tag = "O"
        prev_tag = y[m+1]
        
        
        for u in range(len(tags)):
            pi = T[m][u]
            if (T[m][u]!=0):
                try:
                    # if(tr_params[prev_tag]!=0):
                    a = np.log(tr_params[prev_tag][tags[u]])
                    value = pi+a
                    
                except KeyError:
                    # print("got key error")
                    value = -1000000
                

                # print(pi, prev_tag, tags[u], a, value)
            # print(value)
            if(value>max_y):
                max_y = value
                max_tag = tags[u]
        y[m] = max_tag
        # print(words[m], max_tag)
    return y

def predictions_file(inputFile, outputfile, y):
    f = open(outputfile, 'w+', encoding="utf8")

    finput = open(inputFile, 'r', encoding="utf8")
    lines = finput.readlines()
    
    for i in range(len(lines)):
        if (lines[i].rstrip() != ""):
            word = lines[i].rstrip()
            # print(word, y[i+1])
            f.write(word+" "+y[i]+"\n")
            
        else:
            f.write('\n')
    f.close()
    print('Finished writing to File.')

if __name__ == "__main__":
    # file_em = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/train"
    # em_params, t, w = em.emmissionWithSmoothing(open(file_em, "r", encoding="utf8"), 3)
    
    # tr_params = tr.transition(open(file_em, "r", encoding="utf8"))
    
    
    # tags = list(tr_params)
    
    
    # print(tags)
    # tags.remove("STOP")
    # tags.remove("START")
    # filePath = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/dev.in"
    # y = viterbi( open(filePath, "r", encoding="utf8"), em_params, tr_params, tags, w)
    # # print(y)

    # fileout = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/dev.p3.out"
    # predictions_file(filePath, fileout, y)


    fhw = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/hwtrain.txt"
    em_hw, thw, whw = em.emmissionWithSmoothing(open(fhw, "r", encoding="utf8"), 3)
    
    tr_hw = tr.transition(open(fhw, "r", encoding="utf8"))
    
    tagshw = list(tr_hw)
    print(tagshw)
    tagshw.remove("STOP")
    tagshw.remove("START")
    print("em_hw ",em_hw)
    print("tr_hw ",tr_hw)
    filePath = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/hwin.txt"
    y = viterbi( open(filePath, "r", encoding="utf8"), em_hw, tr_hw, tagshw, whw)
    print(y)

    fileout = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/dev.hw.out"
    predictions_file(filePath, fileout, y)


    
