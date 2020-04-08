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

    trainingList = ["b","e","f","g",
                    "h","i","j","k","l","m","n",
                    "o","p","r","s","t","u",
                    "v","w","x","y","z","cancelalarm","canceltimer",
                    "closedfist","ok","set"]
    trainingData = []
    trainingDataY = []
    trainingData = []
    trainingDataY = []

    reference = "reference.csv"

    image_file = "test.png"
    dir = "output/"
    os.system("./build/examples/openpose/openpose.bin --image_dir samples/ --write_images results --hand --write_json output")

    json_file = dir + (image_file.split("png")[0] + "_keypoints.json")

    ref_file = open(reference,'r')
    ref_reader = csv.reader(ref_file)
    ref = []
    for row in ref_reader:
        for data in row:
            ref.append(data)


    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    data = json_dict["people"][0]
    hand_data = data["hand_right_keypoints_2d"]
    if (len(hand_data)//3 != 21):
        print("not enough features to classify")
    else:
        temp = []
        body_data = data["pose_keypoints_2d"]
        body_3x,body_3y = body_data[3*3],body_data[3*3+1]
        body_4x,body_4y = body_data[4*3],body_data[4*3+1]
        temp.append(distance(body_3x,body_3y,body_4x,body_4y))
        for i in range(21):
            temp.append(hand_data[i*3])
            temp.append(hand_data[i*3+1])

    length_arm = float(row[0])
    ratio = length_arm / relative_length_arm
    base_x = float(row[1])
    base_y = float(row[2])
    new_points.append(base_x)
    new_points.append(base_y)
    for feature in range(1,21):
        feature_x = float(temp[1+feature*2])
        feature_y = float(temp[1+feature*2+1])
        rise = feature_y - base_y
        run = feature_x - base_x
        angle = math.atan2(rise,run)
        segment_length = distance(base_x,base_y,feature_x,feature_y) * ratio
        # print("relative length is %f, segment length is %f" %(relative_points[feature-1],segment_length))
        ##offset by one because relative_points has 20 inputs
        new_rise = math.sin(angle) * segment_length
        new_run = math.cos(angle) * segment_length
        print(new_rise)
        print(new_run)

        new_x = base_x + new_run
        new_y = base_y + new_rise

        new_points.append(new_x)
        new_points.append(new_y)
    base_x = new_points[0]
    base_y = new_points[1]
    for i in range(21):
        new_points[i*2] = float(new_points[i*2] - base_x)
        new_points[i*2+1] = float(new_points[i*2+1] - base_y)


    estimator = joblib.load("%s.pkl"%"linear_svc")


    y_pred = estimator.predict(new_points)
    correct = 0
    incorrect = 0
    for i in range(len(y_pred)):
        if y_pred[i] != testDataY[i]:
            incorrect += 1
            print("incorrect with label %s, guessed %s " %(trainingList[y_pred[i]],trainingList[testDataY[i]]))
        else:
            correct += 1
            print("correct with label %d" %(y_pred[i]))
    print("correctness for linear is %f" %((correct)/(correct+incorrect)))

if __name__ == '__main__':
    main()
