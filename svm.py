from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
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

def main():
    trainingData = []
    trainingDataY = []
    label = "b"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = 0
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(label)


    label = "cancelalarm"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = 1
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(label)

    label = "canceltimer"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = 2
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(label)

    label = "f"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = 3
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(label)

    label = "ok"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = 4
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(label)
        # print(len(temp))
    label = "set"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = 5
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(label)

    label = "wave"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = 6
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append(np.asarray(temp))
        trainingDataY.append(label)

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

    print(trainingDataY)
    newData = trainingData[:int(len(trainingData)*0.8)]
    testData = trainingData[int(len(trainingData)*0.8):]
    newDataY = trainingDataY[:int(len(trainingData)*0.8)]
    testDataY = trainingDataY[int(len(trainingData)*0.8):]

    svclassifier = SVC(kernel = "linear").fit(newData,newDataY)
    rbf_svc = SVC(kernel='rbf', gamma=3).fit(newData, newDataY)
    poly_svc = SVC(kernel='poly', degree=3).fit(newData, newDataY)

    y_pred = svclassifier.predict(testData)
    y_pred1 = rbf_svc.predict(testData)
    y_pred2 = poly_svc.predict(testData)

    correct = 0
    incorrect = 0
    for i in range(len(y_pred)):
        if y_pred[i] != testDataY[i]:
            incorrect += 1
        else:
            correct += 1
    print("correctness for linear is %f" %((correct)/(correct+incorrect)))
    correct = 0
    incorrect = 0
    for i in range(len(y_pred)):
        if y_pred1[i] != testDataY[i]:
            incorrect += 1
        else:
            correct += 1
    print("correctness for gaussian is %f" %((correct)/(correct+incorrect)))
    correct = 0
    incorrect = 0
    for i in range(len(y_pred)):
        if y_pred2[i] != testDataY[i]:
            incorrect += 1
        else:
            correct += 1
    print("correctness for polynomial is %f" %((correct)/(correct+incorrect)))


if __name__ == '__main__':
    main()
