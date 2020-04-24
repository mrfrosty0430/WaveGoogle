import csv
import math

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
    filepath = "u_vs_v_normalized.csv"
    write_file = open(filepath,mode='w')
    csv_writer = csv.writer(write_file)

    input = "u_vs_v.csv"
    read_file = open(input,mode='r')
    csv_reader = csv.reader(read_file)


    reference = "reference.csv"
    ref_file = open(reference,'r')
    ref_reader = csv.reader(ref_file)
    ref = []
    for row in ref_reader:
        for data in row:
            ref.append(data)




    count = 0
    for row in csv_reader:
        new_points = []

        length_arm = float(row[0])
        ratio = length_arm / float(ref[0])
        base_x = float(row[1])
        base_y = float(row[2])
        new_points.append(base_x)
        new_points.append(base_y)
        for feature in range(1,21):
            feature_x = float(row[1+feature*2])
            feature_y = float(row[1+feature*2+1])
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
        new_points.append(int(row[-1]))

        base_x = new_points[0]
        base_y = new_points[1]
        for i in range(21):
            new_points[i*2] = float(new_points[i*2] - base_x)
            new_points[i*2+1] = float(new_points[i*2+1] - base_y)

        count += 1
        csv_writer.writerow(new_points)


    read_file.close()
    write_file.close()







if __name__ == '__main__':
    main()
