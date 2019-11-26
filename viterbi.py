import part3 as tr
import part2 as em
import numpy as np
# model parameters

# em_params, tags = em.emmissionWithSmoothing(file, k)
# b_v(x) = em_params[v][x] (v=tag, x=word)

# tr_params = tr.transition(file)
# a(u, v) = tr_params[v][u]

def viterbi(file, em_params, tr_params, tags):
    words = file.readlines()

    # FORWARD PROPAGATION
    n = len(words)
    print(n)
    T = [[0 for i in range(len(tags))] for i in range(n) ]
    # print(tr_params["STOP"])
    # base case
    for u in range(len(tags)):
        try:
            T[0][u] = -1*np.log(tr_params[tags[u]]["START"]) - np.log(em.getLikelihood(em_params, words[0], tags[u]))
        except KeyError:
            T[0][u] = 0
        except TypeError:
            T[0][u] = 0
    
    for k in range(1, n):
        # print(k)
        if (words[k].rstrip() != ""):
            word = words[k].rstrip().rsplit(' ', 1)
            for v in range(len(tags)):
                find_max = []
                for u in range(len(tags)):
                    
                    try:
                        pi = T[k-1][u]
                        
                        if(tr_params[tags[v]] != 0) :
                            a = -1*np.log(tr_params[tags[v]][tags[u]])
                            b = em.getLikelihood(em_params, word[0], tags[v])
                            if(b!=0):
                                bx = -1*np.log(b)
                            
                            value = pi+a+bx
                            
                        else: 
                            value = 0
                    except KeyError:
                        value = 0
                    find_max.append(value)
                
                T[k][v] = max(find_max) 
    
    # print("Finished forward propagation")
    # print(*T, sep = "\n") 
    # BACKWARD PROPAGATION

    # find last layer with no zeros
    z = n-1
    y = [0 for x in range(n-1)]
    max_y = 0
    max_tag = ""
    
    last_layer = T[n-2]
    max_value = 0

    for u in range(len(tags)):
        try:
            x = (T[n-2][u]-np.log(tr_params["STOP"][tags[u]]))
        except KeyError:
            x = 0
        if (x>max_value):
            max_value = x
            max_tag = tags[u]
    
    # max_tag = tags[max_index]
    y[n-2] = max_tag
    



    for m in range(n-3, -1, -1):
        max_y = 0
        max_tag = ""
        prev_tag = y[m+1]
        

        for u in range(len(tags)-1):
            pi = T[m][u]
            try:
                if(tr_params[prev_tag]!=0):
                    a = -1*np.log(tr_params[prev_tag][tags[u]])
                else:
                    a=0
            except KeyError:
                a = 0
            value = pi+a
            # print(words[m], tags[u], value, a)
            if(value>max_y):
                max_y = value
                max_tag = tags[u]
        y[m] = max_tag

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
    file_em = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/train"
    em_params, tagCount = em.emmissionWithSmoothing(open(file_em, "r", encoding="utf8"), 200)
    
    tr_params = tr.transition(open(file_em, "r", encoding="utf8"))
    tags = list(tr_params)
    
    
    print(tags)
    tags.remove("STOP")
    filePath = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/dev.in"
    y = viterbi( open(filePath, "r", encoding="utf8"), em_params, tr_params, tags)
    print(y)

    fileout = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/dev.p3.out"
    predictions_file(filePath, fileout, y)

    
