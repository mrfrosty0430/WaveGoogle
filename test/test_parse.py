import json
from enum import IntEnum
import math
import csv
import os

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

class Hand (IntEnum):
    BASE_X = 0
    BASE_Y = 1
    BASE_AUTO = 2

    PINKY_BASE_X = 3
    PINKY_BASE_Y = 4
    PINKY_BASE_AUTO = 5

    PINKY_ONE_X = 6
    PINKY_ONE_Y = 7
    PINKY_ONE_AUTO = 8

    PINKY_TWO_X = 9
    PINKY_TWO_Y = 10
    PINKY_TWO_AUTO = 11

    PINKY_TIP_X = 12
    PINKY_TIP_Y = 13
    PINKY_TIP_AUTO = 14

    RING_BASE_X = 15
    RING_BASE_Y = 16
    RING_BASE_AUTO = 17

    RING_ONE_X = 18
    RING_ONE_Y = 19
    RING_ONE_AUTO = 20

    RING_TWO_X = 21
    RING_TWO_Y = 22
    RING_TWO_AUTO = 23

    RING_TIP_X = 24
    RING_TIP_Y = 25
    RING_TIP_AUTO = 26

    MIDDLE_BASE_X = 27
    MIDDLE_BASE_Y = 28
    MIDDLE_BASE_Z = 29

    MIDDLE_ONE_X = 30
    MIDDLE_ONE_Y = 31
    MIDDLE_ONE_AUTO = 32

    MIDDLE_TWO_X = 33
    MIDDLE_TWO_Y = 34
    MIDDLE_TWO_AUTO = 35

    MIDDLE_TIP_X = 36
    MIDDLE_TIP_Y = 37
    MIDDLE_TIP_AUTO = 38

    POINTER_BASE_X = 39
    POINTER_BASE_Y = 40
    POINTER_BASE_AUTO = 41

    POINTER_ONE_X = 42
    POINTER_ONE_Y = 43
    POINTER_ONE_AUTO = 44

    POINTER_TWO_X = 45
    POINTER_TWO_Y = 46
    POINTER_TWO_AUTO = 47

    POINTER_TIP_X = 48
    POINTER_TIP_Y = 49
    POINTER_TIP_AUTO = 50

    THUMB_BASE_X = 51
    THUMB_BASE_Y = 52
    THUMB_BASE_AUTO = 53

    THUMB_ONE_X = 54
    THUMB_ONE_Y = 55
    THUMB_ONE_AUTO = 56

    THUMB_TWO_X = 57
    THUMB_TWO_Y = 58
    THUMB_TWO_AUTO = 59

    THUMB_TIP_X = 60
    THUMB_TIP_Y = 61
    THUMB_TIP_AUTO = 62

def main():

    trainingList = ["b","e","f","g",
                    "h","i","j","k","l","m","n",
                    "o","p","r","s","t","u",
                    "v","w","x","y","z","cancelalarm","canceltimer",
                    "closedfist","ok","set"]


    for training in trainingList:
        # training = "wave"
        dir = "output/" + training
        filepath = "csv/" + training + ".csv"
        employee_file = open (filepath, mode='w',)
        csv_writer = csv.writer(employee_file)
        for filename in os.listdir(dir):
            try:
                print(os.path.join(dir,filename))
                with open(os.path.join(dir,filename), 'r') as f:
                    json_dict = json.load(f)

                data = json_dict["people"][0]
                hand_data = data["hand_right_keypoints_2d"]


            # print(len(hand_data))
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

                    csv_writer.writerow(temp)
            except:
                pass
        employee_file.close()

    #
    #
    # print("num features: %d" %(len(hand_data)//3))
    #


if __name__ == '__main__':
    main()
