import os
import csv


def main():

    trainingList = ["u","v"]

    filepath = "u_vs_v.csv"
    write_file = open(filepath, mode='w')
    csv_writer = csv.writer(write_file)


    count = 0;
    for label in trainingList:

        filepath = "csv/" + label + ".csv"
        employee_file = open (filepath, mode='r')
        csv_reader = csv.reader(employee_file)

        for row in csv_reader:
            temp = []
            for data in row:
                temp.append(data)
            temp.append(count)
            csv_writer.writerow(temp)
        employee_file.close()
        count += 1
    write_file.close()



if __name__ == '__main__':
    main()
