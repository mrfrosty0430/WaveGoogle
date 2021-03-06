import os
import csv


def main():

    trainingList = ['a', 'b', 'cancelalarm', 'canceltimer', 'closedfist', 'e', 'eight',
                    'f', 'five', 'four', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'nine',
                    'o', 'ok', 'one', 'p', 'r', 's', 'set', 'settimer', 'seven', 'six',
                    'stop', 't', 'three', 'two', 'u', 'v', 'w', 'x', 'y', 'z', 'zero']

    filepath = "alldata.csv"
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
