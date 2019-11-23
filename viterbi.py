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
    T = [[0 for i in range(len(tags))] for i in range(n+2) ]
   
    
    for k in range(0, n+2):
        
            for v in range(len(tags)):
                # base case : START
                if (k==0) and (tags[v]=="START"):
                    T[k][v] = 0
                if (k==0) and (tags[v]!="START"):
                    T[k][v] = 0
                # stop case : STOP
                if (k==n+1) and (tags[v]=="STOP"):
                    find_max = []
                    for u in range(len(tags)):
                        try:
                            # print(np.log(tr_params["STOP"][tags[u]]))
                            value = T[n][u] - np.log(tr_params["STOP"][tags[u]])
                        except KeyError:
                            value = 0
                        find_max.append(value)
                    T[k][v] = max(find_max)
                if(k!=0) and (k!=n+1):
                    if (words[k-1].rstrip() != ""):
                        word = words[k-1].rstrip().rsplit(' ', 1)
                        
                        find_max = []
                        for u in range(len(tags)):
                            
                            try:
                                pi = T[k-1][u]
                                
                                if(tr_params[tags[v]] != 0) :
                                    a = -1*np.log(tr_params[tags[v]][tags[u]])
                                    # print("a ",a)
                                    b = em.getLikelihood(em_params, word[0], tags[v])
                                    if(b!=0):
                                        bx = -1*np.log(b)
                                    # print("b ",b)
                                    # b = em_params[tags[v]][word[0]]
                                    # if(a!=np.inf) and (b!=np.inf):
                                    value = pi+a+bx
                                    # print(a, b, value)
                                    
                                else: 
                                    value = 0
                            except KeyError:
                                value = 0
                            # value = T[k-1][u]*tr_params[v][u]*em_params[v][word[0]]
                            find_max.append(value)
                        # print(find_max)
                        T[k][v] = max(find_max) 
    
    # print(T)
    # BACKWARD PROPAGATION
    y = [0 for x in range(n+2)]
    max_y = 0
    max_tag = ""
    
    last_layer = T[n+1]
    max_index = last_layer.index(max(last_layer))
    max_tag = tags[max_index]
    y[len(words)] = max_tag
    # print(y)
    y[len(words)+1] = "STOP"



    for m in range(n-1, -1, -1):
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
            if(value>=max_y):
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
            f.write(word+" "+y[i+1]+"\n")
            
        else:
            f.write('\n')
    f.close()
    print('Finished writing to File.')

if __name__ == "__main__":
    file_em = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/train"
    em_params, tagCount = em.emmissionWithSmoothing(open(file_em, "r", encoding="utf8"), 200)
    
    tr_params = tr.transition(open(file_em, "r", encoding="utf8"))
    tags = list(tr_params)
    
    tags.insert(0, "START")
    tags.insert(len(tags), "STOP")
    filePath = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/dev.in"
    y = viterbi( open(filePath, "r", encoding="utf8"), em_params, tr_params, tags)
    print(y)

    fileout = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/dev.p3.out"
    predictions_file(filePath, fileout, y)

    
