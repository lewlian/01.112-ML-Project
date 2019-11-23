import part3 as tr
import part2 as em

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
                    T[k][v] = 1
                if (k==0) and (tags[v]!="START"):
                    T[k][v] = 0
                # stop case : STOP
                if (k==n+1) and (tags[v]=="STOP"):
                    find_max = []
                    for u in range(len(tags)):
                        try:
                            value = T[n][u] * tr_params["STOP"][tags[u]]
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
                                a = tr_params[tags[v]][tags[u]]
                                b = em_params[tags[v]][word[0]]
                                value = pi*a*b
                            except KeyError:
                                value = 0
                            # value = T[k-1][u]*tr_params[v][u]*em_params[v][word[0]]
                            find_max.append(value)
                        T[k][v] = max(find_max) 
    
    

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
                a = tr_params[prev_tag][tags[u]]
            except KeyError:
                a = 0
            value = pi*a
            if(value>=max_y):
                max_y = value
                max_tag = tags[u]
        y[m] = max_tag

    return y

if __name__ == "__main__":
    file_em = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/train"
    em_params, tagCount = em.emmissionWithSmoothing(open(file_em, "r", encoding="utf8"), 200)
    # print(len(list(em_params)))
    
    # print(lentags)
    tr_params = tr.transition(open(file_em, "r", encoding="utf8"))
    tags = list(tr_params)
    print(tags)
    # print(len(list(tr_params)))
    filePath = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/dev.in"
    y = viterbi( open(filePath, "r", encoding="utf8"), em_params, tr_params, tags)
    # print(y)

    
