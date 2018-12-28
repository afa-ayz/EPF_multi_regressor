
from numpy import genfromtxt
import numpy as np
from datetime import datetime

dictionary = {'VIC1':0, 'NSW1':1, 'QLD1':2, 'SA1':3, 'TAS1': 4}


def decode_date(date):
    exact_time = date.split(" ")[-1]
    exact_time = int(exact_time[:2]) * 2 + int(int(exact_time[3:5]) / 30)
    year, month, day = int(date[:4]), int(date[5:7]), int(date[8:10])
    week = datetime(year,month,day).weekday()
    return month, day, week + 1, exact_time


def load_data(csv_path):
    my_data = genfromtxt(csv_path, delimiter=',',dtype=str)[1:]
    data = []
    for each_record in my_data:
        temp = []

        # region
        temp.append(dictionary[each_record[0]])

        # time
        month, day, week, exact_time = decode_date(each_record[1])
        temp.append(month)
        temp.append(day)
        temp.append(week)
        temp.append(exact_time)

        # demand
        temp.append(float(each_record[-3]))

        # price
        temp.append(float(each_record[-2]))


        data.append(temp)

    return np.array(data)

