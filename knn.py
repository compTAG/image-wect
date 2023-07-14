import numpy as np
import json
import random
import pandas as pd

from sklearn.metrics import accuracy_score

from imagewect.wect import distance_between_wects_multidirectional

class KNN:
    def __init__(self, X, y, k=3) -> None:
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        preds = np.zeros((len(X_test)))
        for j, tp in enumerate(X_test):
            distances = np.zeros((len(self.X),))
            for i, x in enumerate(self.X):
                distances[i] = distance_between_wects_multidirectional(tp, x, -43, 43)
            sort_inds = distances.argsort()
            y = self.y[sort_inds]
            values, counts = np.unique(y[:self.k], return_counts=True)
            ind = np.argmax(counts)
            preds[j] = values[ind]
            print(f"predicted {j}/{len(X_test)}")
        return preds
    

def train_test_split(shape1, shape2, independent=False):
    inds1, inds2 = list(range(250)), list(range(250))
    random.shuffle(inds1)
    random.shuffle(inds2)
    test_inds1, train_inds1 = inds1[:25], inds1[25:125]
    test_inds2, train_inds2 = inds2[:25], inds2[25:125]
    if independent:
        test_inds2, train_inds2 = inds1[125:150], inds1[150:250]
    
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in train_inds1:
        try:
            X_train.append(shape1[f"{i}"])
        except KeyError:
            X_train.append(shape1[i])
        y_train.append(0)
    for i in test_inds1:
        try:
            X_test.append(shape1[f"{i}"])
        except KeyError:
            X_test.append(shape1[i])
        y_test.append(0)
    for i in train_inds2:
        try:
            X_train.append(shape2[f"{i}"])
        except KeyError:
            X_train.append(shape2[i])
        y_train.append(1)
    for i in test_inds2:
        try:
            X_test.append(shape2[f"{i}"])
        except KeyError:
            X_test.append(shape2[i])
        y_test.append(1)
    
    return X_train, np.array(y_train), X_test, np.array(y_test)

def main():

    # shapes = ["annulus", "circle", "clusters", "square_annulus", "swiss_cheese", "tetris", "square"]
    shapes = ["swiss_cheese"]
    # distributions = ["uniform", "n17", "n25", "n50"]
    distributions = ["uniform"]
    dirs = "8dir"

    for shape in shapes:
        for dist in distributions:
            with open(f"data/wects_proper/MAXfe/8dir/{shape}/{dist}.json") as f1:
                shape1 = json.load(f1)

            with open(f"data/wects_proper/MAXfe/8dir/{shape}/uniform.json") as f2:
                shape2 = json.load(f2)

        
            accuracies = np.zeros((100,))
            ind = False
            if dist == "uniform":
                ind = True
            for i in range(0,100):
                X_train, y_train, X_test, y_test = train_test_split(shape1, shape2, ind)

                model = KNN(X_train, y_train, k=5)
                preds = model.predict(X_test)
                accuracies[i] = accuracy_score(y_test, preds)
                print(f"finished trial {i+1}/100, accuracy", accuracies[i])
            
            print(f"{shape} {dist} vs. {shape} uniform 8dir max")
            with open("data/results_knn/vs_own_uniform.txt", "a") as outfile:
                outfile.write(f"{shape},max,{dist},{np.mean(accuracies)},{np.std(accuracies)}\n")


main()
