import csv
import math

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():

    trainingList = ["a","b","c","d","e","f","g",
                    "h","i","j","k","l","m","n",
                    "o","p","q","r","s","t","u",
                    "v","w","x","y","z","cancelalarm","canceltimer",
                    "closedfist","ok","set"]
    for label in trainingList:
        # label = "wave"
        dir = "output/" + label
        filepath = "csv/" + label + ".csv"
        employee_file = open (filepath, mode='r')
        csv_reader = csv.reader(employee_file)

        outdir = "processed/" + label
        filepath = "processed/" + label + ".csv"
        write_file = open(filepath, mode='w'
        csv_writer = csv.writer(write_file)

        count = 0
        relative_points = []
        for row in csv_reader:
            new_points = []
            print("row is %d" %(count))
            if count == 0:
                #--------all hands relative to this one-------------#

                base_x = float(row[0])
                base_y = float(row[1])
                new_points.append(base_x)
                new_points.append(base_y)

                for feature in range(1,21):
                    feature_x = float(row[feature*2])
                    feature_y = float(row[feature*2+1])
                    new_points.append(feature_x)
                    new_points.append(feature_y)
                    relative_points.append(distance(base_x,base_y,feature_x,feature_y))


                for i in range(21):
                    print(i)
                    new_points[i*2] = new_points[i*2] - base_x
                    new_points[i*2+1] = new_points[i*2+1] - base_y

                print(relative_points)
            else:

                base_x = float(row[0])
                base_y = float(row[1])
                new_points.append(base_x)
                new_points.append(base_y)
                for feature in range(1,21):
                    feature_x = float(row[feature*2])
                    feature_y = float(row[feature*2+1])

                    rise = feature_y - base_y
                    run = feature_x - base_x
                    angle = math.atan2(rise,run)
                    segment_length = distance(base_x,base_y,feature_x,feature_y)
                    print("relative length is %f, segment length is %f" %(relative_points[feature-1],segment_length))
                    ##offset by one because relative_points has 20 inputs
                    new_rise = math.sin(angle) * relative_points[feature-1]
                    new_run = math.cos(angle) * relative_points[feature-1]
                    print(new_rise)
                    print(new_run)

                    new_x = base_x + new_run
                    new_y = base_y + new_rise

                    new_points.append(new_x)
                    new_points.append(new_y)


                print(new_points)
                base_x = new_points[0]
                base_y = new_points[1]
                for i in range(21):
                    new_points[i*2] = float(new_points[i*2] - base_x)
                    new_points[i*2+1] = float(new_points[i*2+1] - base_y)


            count += 1
            csv_writer.writerow(new_points)





        employee_file.close()



























if __name__ == '__main__':
    main()
