from collections import defaultdict
import sys
import json
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



if __name__ == "__main__":
    
    filePath = "./EN/train"
    mle = transition(
        open(filePath, "r", encoding="utf8"))
    # mle consists of all transition parameters
    
