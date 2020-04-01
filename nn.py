import torch
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import datetime
import random
import csv

def main():


    trainingData = []
    label = "b"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = [1,0,0,0,0,0,0]
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append((np.asarray((temp)),np.asarray(label)))

    label = "cancelalarm"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = [0,1,0,0,0,0,0]
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append((np.asarray((temp)),np.asarray(label)))

    label = "canceltimer"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = [0,0,1,0,0,0,0]
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append((np.asarray((temp)),np.asarray(label)))

    label = "f"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = [0,0,0,1,0,0,0]
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append((np.asarray((temp)),np.asarray(label)))

    label = "ok"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = [0,0,0,0,1,0,0]
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append((np.asarray((temp)),np.asarray(label)))
        # print(len(temp))
    label = "set"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = [0,0,0,0,0,1,0]
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append((np.asarray((temp)),np.asarray(label)))

    label = "wave"
    filepath = "processed/" + label + ".csv"
    employee_file = open (filepath, mode='r')
    csv_reader = csv.reader(employee_file)
    label = [0,0,0,0,0,0,1]
    for row in csv_reader:
        temp = []
        for elem in row:
            temp.append(float(elem))
        trainingData.append((np.asarray((temp)),np.asarray(label)))

    # print(trainingData)
    random.shuffle(trainingData)
    newData = trainingData[:int(len(trainingData)*0.8)]
    testData = trainingData[int(len(trainingData)*0.8):]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                  ])
    trainloader = torch.utils.data.DataLoader(newData, batch_size=1, shuffle=True)

    hiddenlayer = 3
    # hiddenlayer2 = 32
    model = nn.Sequential(nn.Linear(42, hiddenlayer),
                          nn.Sigmoid(),
                          nn.Linear(hiddenlayer,7),


                          # nn.Sigmoid())
                          nn.LogSoftmax(dim=1))
    # Define the loss
    criterion = nn.NLLLoss()
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 10
    for e in range(epochs):
        running_loss = 0
        # print(len(trainloader))
        for feature, labels in trainloader:

            optimizer.zero_grad()

            output = model(feature.float())
            # if output.argmax() == labels.argmax():
                # print("fuck ya")
            # else:
                # print("FUCKFUCKFUCK")
            # print(output)
            loss = criterion(output, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Training loss at epoch %d: {running_loss/len(trainloader)}" %(e))


    #
    #
    testloader = torch.utils.data.DataLoader(testData, batch_size=1, shuffle=True)

    correct = 0
    incorrect = 0
    for feature, labels in testloader:
        optimizer.zero_grad()
        output = model(feature.float())
        if (output.argmax() == labels.argmax()):
            correct += 1

        else:
            incorrect +=  1
            print(output.argmax(),labels.argmax());
            if (output.argmax() == 0 and labels.argmax() == 3):
                print(output)
            if (output.argmax() == 0):
                if (abs(output[0][0]-output[0][6])<2):
                    if (labels.argmax() == 6):
                        correct += 1
                    else:
                        incorrect += 1
                else:
                    incorrect += 1
            else:

                incorrect += 1
            print(output)
    print("correct is %d" % (correct))
    print("incorrect is %d" % (incorrect))
    print("correctness is %f" %((correct/(correct+incorrect))*100))
    #



if __name__ == '__main__':
    main()
