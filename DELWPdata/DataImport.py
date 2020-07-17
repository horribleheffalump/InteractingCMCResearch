import numpy as np
import pandas as pd
from datetime import date


def parse(t):
    string_ = str(t)
    try:
        return date(int(string_[6:10]), int(string_[3:5]), int(string_[0:2]))
    except ValueError:
        return np.NaN


def import_data(filename, title, col, start, drop_q=[255, 180, 153, 160]):
    headers = ['Datetime', title, 'quality']
    dtypes = {'Datetime': 'str', title: 'float', 'quality': 'int'}
    parse_dates = ['Datetime']
    data = pd.read_csv(filename, sep=',', usecols=[0, col, col + 1], skiprows=start, header=None, names=headers,
                       dtype=dtypes, parse_dates=parse_dates, date_parser=parse)
    data.index = data.Datetime
    data.drop(data[data['quality'].isin(drop_q)].index, inplace=True)
    data.drop(['Datetime', 'quality'], axis=1, inplace=True)
    data.dropna(inplace=True)
    return data


def import_rainfall(filename, start):
    headers = ['year', 'month', 'day', 'Rainfall']
    dtypes = {'year': 'int', 'month': 'int', 'day': 'int', 'Rainfall': 'float'}
    data = pd.read_csv(filename, sep=',', usecols=[2, 3, 4, 5], skiprows=start, header=None, names=headers,
                       dtype=dtypes)
    data['Datetime'] = list(
        map(lambda x: date(x[0], x[1], x[2]), zip(data['year'].values, data['month'].values, data['day'].values)))
    data.index = data.Datetime
    data.drop(['year', 'month', 'day', 'Datetime'], axis=1, inplace=True)
    data.dropna(inplace=True)
    return data
