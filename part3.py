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

    # FIRST LINE
    first_line = lines[0]
    w = first_line.rstrip().rsplit(' ', 1)
    first_tag = w[1]
    tagToWordDictionary[first_tag]["START"] += 1
    tagCount[first_tag] += 1
    # print(first_line)

    for i in range(1, len(lines)):
        line = lines[i]
        # print(line)
        previous_line = lines[i-1]

        # STOP CASE : line = "", prev_line = "tag"
        if (line.rstrip() == ""):
            w_i1 = previous_line.rstrip().rsplit(' ', 1)
            
            tag_i1 = w_i1[1]
            tagToWordDictionary["STOP"][tag_i1] += 1
            tagCount["STOP"] += 1
        
        # START CASE : prev_line = "", line = "tag"
        if (previous_line.rstrip() == ""):
            w_i = line.rstrip().rsplit(' ', 1)

            tag_i = w_i[1]
            tagToWordDictionary[tag_i]["START"] += 1
            tagCount[tag_i] += 1

        if (line.rstrip() != "") and (previous_line.rstrip() != ""):
            #   strip all trailing spaces first and then split by white spaces limit to 1
            w_i = line.rstrip().rsplit(' ', 1)
            w_i1 = previous_line.rstrip().rsplit(' ', 1)
            
            tag_i = w_i[1]
            tag_i1 = w_i1[1]
            
            # case 3 = other words
            
            tagToWordDictionary[tag_i][tag_i1] += 1
            tagCount[tag_i] += 1

    # f = open("tags.txt", "w")
    # f.write(json.dumps(tagCount))
    # f.close()

    # f = open("tags_dict.txt", "w")
    # f.write(json.dumps(tagToWordDictionary))
    # f.close()

    mle = defaultdict(int)

    #calculate transition probs
    # print("tagCount ",str(tagCount))
    # print("tagToWordDictionary ",str(tagToWordDictionary))
    for tag in tagCount:
        #   Create a dictionary for each tag which is y-values
        mle[tag] = {}
        #   list(tagToWordDictionary[tag]) returns us ALL the words (all x-values)
        
        for tag_i_minus_1 in list(tagToWordDictionary[tag]):
            mle[tag][tag_i_minus_1] = float(
                tagToWordDictionary[tag][tag_i_minus_1]) / (tagCount[tag])

    return mle



if __name__ == "__main__":
    
    filePath = "/Users/nashitaabd/Documents/GitHub/01.112-ML-Project/EN/train"
    mle = transition(
        open(filePath, "r", encoding="utf8"))
    
