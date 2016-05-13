## D number of documents
## W number of words in the vocabulary
## NNZ number of nonzero counts in the bag-of-words

## {docID : {wordID : count}}
import numpy as np
import math
from decimal import Decimal

myDict = {}

with open('docword.nips.txt') as f:
    for _ in xrange(3):
        next(f)
    for line in f:
        line = line.strip().split(" ")
        key = int(line[0])
        if myDict.has_key(key) is False:
            myDict[key] = {}
        myDict[key][int(line[1])] = int(line[2])

# print myDict[1]
## Assign initial probability for each topic. We have 30 topic in total
topicP = {}
for i in range(0, 30):
    topicP[i] = 1.0 / 30

## Assign each doucment to a topic
topicAssign = {}
for i in range(0, 30):
    topicAssign[i] = []
    for j in range(1, 51):
        topicAssign[i].append(50 * i + j)

# total word count in a topic
topicWordTotal = []
for i in range(0, 30):
    temp = 0
    for j in range(1, 51):
        temp += sum(myDict[50 * i + j].values())
    topicWordTotal.append(temp);

## count each word occurance in a topic
wordInTopicCount = {}
for i in range(0, 30):
    wordInTopicCount[i] = {}
    for j in range(1, 51):
        myWord = myDict[50 * i + j]
        for key in myWord.keys():
            if key in wordInTopicCount[i]:
                wordInTopicCount[i][key] += myWord[key]
            else:
                wordInTopicCount[i][key] = myWord[key]

## calculate each word probability in each topic
test = {}
for j in range(0, 30):
    test[j] = {}
    for i in range(1, 51):
        myWord = myDict[50 * j + i]
        for wordID in myWord.keys():
            exp = myWord[wordID]
            pjk = wordInTopicCount[j][wordID] * 1.0 / topicWordTotal[j]
            temp = math.log(pjk) * exp
            test[j][wordID][i] = temp

## calculate numerator
numerator = []
for i in range(0, 30):
    temp = 0
    for j in range(1, 51):
        myWord = myDict[50 * i + j]
        for key in myWord.keys():
            temp += test[i][key]
    numerator.append(temp + math.log(1.0 / 30))

maximum + 1.0 / 30t
#print numerator

denominator = []
