#Euclidean Distance
#n = number of dimensions of data
#i = 1 goes up to n
#sqrt(Σ(Ai-Pi)^2)

#q = (1,3)
#p=(2,5)
#=sqrt( (1 - 2)^2 + (3 - 5)^2 )

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
##plot1 = [1,3]
##plot2 = [2,5]
##
##euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2 )
##
##print(euclidean_distance)


##[[plt.scatter(ii[0],ii[1], s=100, color = i) for ii in dataset[i]] for i in dataset]
##plt.scatter(new_features[0],new_features[1])
##plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warning('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2 )
            #euclidean_distance = np.sqrt( np.sum((np.array(features) - np.array(predict))**2))
            #below function is the fastest running algoithm for big data
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])


    votes = [i[1]for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k



            
    return vote_result, confidence

accuracies = []

for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.webarchive')
    df.replace('?',-99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    #replaces integers treated as strings to type float dor dataset into list of lists
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    #first 20 percent of the data
    train_data = full_data[:-int(test_size * len(full_data))]
    #last 20 percent of the data
    test_data = full_data[-int(test_size * len(full_data)):]


    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])



    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1


            total += 1

    accuracies.append(correct/total)

print(sum(accuracies) / len(accuracies))

