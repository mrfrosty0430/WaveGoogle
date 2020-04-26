from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.svm import SVC
import torch
import os
import copy
import numpy as np
import random
import csv
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pickle

def main():

    i = 0;
    trainingList = ['a', 'b', 'cancelalarm', 'canceltimer', 'closedfist', 'e', 'eight',
                    'f', 'five', 'four', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'nine',
                    'o', 'ok', 'one', 'p', 'r', 's', 'set', 'settimer', 'seven', 'six',
                    'stop', 't', 'three', 'two', 'u', 'v', 'w', 'x', 'y', 'z', 'zero']

    u_vs_vList = ["u","v"]
    b_vs_fList = ["b","f"]
    p_vs_setList = ["p","set"]
    r_vs_xList = ["r","x"]

#
    trainingData = []
    trainingDataY = []

    filepath = "alldata_normalized.csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    for row in csv_reader:
        temp = []
        data = row[:-1]
        for elem in data:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(int(row[-1]))


    testData = []
    testDataY = []
    filepath = "test/testdata_normalized.csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    for row in csv_reader:
        temp = []
        data = row[:-1]
        for elem in data:
            temp.append(float(elem))
        testData.append(np.asarray(temp))
        testDataY.append(int(row[-1]))



    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    trainingData,trainingDataY = shuffle_in_unison(np.asarray(trainingData),np.asarray(trainingDataY))

    linear_svc = SVC(kernel = "linear").fit(trainingData,trainingDataY)
    rbf_svc = SVC(kernel='rbf', gamma=3).fit(trainingData,trainingDataY)
    poly_svc = SVC(kernel='poly', degree=7).fit(trainingData,trainingDataY)

    # y_pred = linear_svc.predict(testData)
    # y_pred1 = rbf_svc.predict(testData)
    # y_pred2 = poly_svc.predict(testData)
    # joblib.dump(linear_svc,"%s.pkl" % ("linear_svc"))
    correct = 0
    incorrect = 0

#
    estimator = joblib.load("%s.pkl"%"linear_svc")
    y_pred = estimator.predict(testData)
    for i in range(len(y_pred)):
        print("on label %s " %trainingList[testDataY[i]],end="")
        if trainingList[y_pred[i]] == "f":
            estimator = joblib.load("%s.pkl"%"b_vs_f_svc")
            tempTestData = [testData[i]]
            y_new_pred = estimator.predict(tempTestData)
            if (b_vs_fList[y_new_pred[0]] != trainingList[testDataY[i]]):
                incorrect += 1
                print("incorrect with label %s, first predicted f, then guessed %s " %(trainingList[testDataY[i]],b_vs_fList[y_new_pred[0]]))
            else:
                correct += 1
                print("correct with label %s" %(b_vs_fList[y_new_pred[0]]))

        elif trainingList[y_pred[i]] == "p":
            estimator = joblib.load("%s.pkl"%"p_vs_set_svc")
            tempTestData = [testData[i]]
            y_new_pred = estimator.predict(tempTestData)
            if (p_vs_setList[y_new_pred[0]] != trainingList[testDataY[i]]):
                incorrect += 1
                print("incorrect with label %s, first predicted p, then guessed %s " %(trainingList[testDataY[i]],p_vs_setList[y_new_pred[0]]))
            else:
                correct += 1
                print("correct with label %s" %(p_vs_setList[y_new_pred[0]]))

        elif trainingList[y_pred[i]] == "x":
            estimator = joblib.load("%s.pkl"%"r_vs_x_svc")
            tempTestData = [testData[i]]
            y_new_pred = estimator.predict(tempTestData)
            if (r_vs_xList[y_new_pred[0]] != trainingList[testDataY[i]]):
                incorrect += 1
                print("incorrect with label %s, first predicted x, then guessed %s " %(trainingList[testDataY[i]],r_vs_xList[y_new_pred[0]]))
            else:
                correct += 1
                print("correct with label %s" %(r_vs_xList[y_new_pred[0]]))

        else:
            if y_pred[i] != testDataY[i]:

                incorrect += 1
                print("incorrect with label %s, guessed %s " %(trainingList[testDataY[i]],trainingList[y_pred[i]]))
            else:
                correct += 1
                print("correct with label %s" %(trainingList[y_pred[i]]))

    print("testing %d images" % (len(y_pred)))
    print("correctness for linear is %f" %((correct)/(correct+incorrect)))

    # correct = 0
    # incorrect = 0
    # for i in range(len(y_pred)):
    #     if y_pred1[i] != testDataY[i]:
    #         incorrect += 1
    #     else:
    #         correct += 1
    # print("correctness for gaussian is %f" %((correct)/(correct+incorrect)))
    # correct = 0
    # incorrect = 0
    # for i in range(len(y_pred)):
    #     if y_pred2[i] != testDataY[i]:
    #         incorrect += 1
    #     else:
    #         correct += 1
    # print("correctness for polynomial is %f" %((correct)/(correct+incorrect)))



if __name__ == '__main__':
    main()
