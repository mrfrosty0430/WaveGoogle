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
    trainingList = ["u","v"]
    trainingData = []
    trainingDataY = []

    filepath = "u_vs_v_normalized.csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    for row in csv_reader:
        temp = []
        data = row[:-1]
        for elem in data:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(int(row[-1]))


    # testData = []
    # testDataY = []
    # filepath = "test/testdata_normalized.csv"
    # employee_file = open (filepath, mode='r')
    # csv_reader = csv.reader(employee_file)
    # for row in csv_reader:
    #     temp = []
    #     data = row[:-1]
    #     for elem in data:
    #         temp.append(float(elem))
    #     testData.append(np.asarray(temp))
    #     testDataY.append(int(row[-1]))
    #


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
    # rbf_svc = SVC(kernel='rbf', gamma=3).fit(trainingData,trainingDataY)
    # poly_svc = SVC(kernel='poly', degree=7).fit(trainingData,trainingDataY)
    #
    # y_pred = linear_svc.predict(testData)
    # y_pred1 = rbf_svc.predict(testData)
    # y_pred2 = poly_svc.predict(testData)
    #
    # correct = 0
    # incorrect = 0
    # for i in range(len(y_pred)):
    #     if y_pred[i] != testDataY[i]:
    #         incorrect += 1
    #         print("incorrect with label %s, guessed %s " %(trainingList[y_pred[i]],trainingList[testDataY[i]]))
    #     else:
    #         correct += 1
    #         print("correct with label %d" %(y_pred[i]))
    # print("correctness for linear is %f" %((correct)/(correct+incorrect)))
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
    joblib.dump(linear_svc,"%s.pkl" % ("u_vs_v_svc"))


if __name__ == '__main__':
    main()
