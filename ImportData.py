import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

def parse(t):
    string_ = str(t)
    try:
        return date(int(string_[6:10]), int(string_[3:5]), int(string_[0:2]))
    except:
        return np.NaN

filename = "D:\\Наука\\_Статьи\\__в работе\\water\\data\\Eildon exit\\405203_all_daily.csv"
#filename = "D:\\Наука\\_Статьи\\__в работе\\water\\data\\GulburnWeir - rainfall levels discharge\\405253.csv"


def import_data(filename, title, col, start, drop_q = [255, 180, 153, 160]):
    headers = ['Datetime', title, 'quality']
    dtypes = {'Datetime': 'str', title: 'float', 'quality': 'int'}
    parse_dates = ['Datetime']
    data = pd.read_csv(filename, sep=',', usecols=[0, col, col+1], skiprows=start, header=None, names=headers, dtype=dtypes, parse_dates=parse_dates, date_parser=parse)
    data.index = data.Datetime
    data.drop(data[data['quality'].isin(drop_q)].index, inplace=True)
    data.drop(['Datetime', 'quality'], axis=1, inplace=True)
    data.dropna(inplace=True)
    return data

datasets = []

datasets.append((import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\Eildon head\\405258.csv", "Eildon-L", 1, 15), 'tab:orange'))
#datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\GoulburnKirwans\\405282.csv", "GoulburnKirwans-L", 1, 15))
datasets.append((import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\GoulburnWeir\\405259.csv", "GoulburnWeir-L", 1, 18), 'tab:red'))
datasets.append((import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\Eildon exit\\405203.csv", "Eildon-D", 3, 18), 'tab:cyan'))
datasets.append((import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\StuartMurrey\\405700.csv", "StuartMurrey-D", 3, 15), 'tab:green'))
datasets.append((import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\GoulburnRiver\\405253.csv", "GoulburnRiver-D", 3, 18), 'tab:olive'))
datasets.append((import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\EastMain\\405704.csv", "EastMain-D", 1, 25), 'tab:gray'))
#datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\GoulburnMurchison\\405200.csv", "GoulburnMurchison-D", 5, 25))
datasets.append((import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\CattanachWaranga\\405702.csv", "Cattanach-D", 3, 25), 'tab:blue'))




data = pd.concat([p[0] for p in datasets], axis=1)

data.drop(data[data.index < pd.Timestamp(2018, 10, 1)].index, inplace=True)

#sns.pairplot(data, palette="YlGnBu")
#g.savefig("pairplot.png")



#colors_l = ['tab:orange', 'tab:red', 'tab:purple']
levels = [c for c in list(data.columns) if '-L' in c]

#colors_d = ['tab:blue', 'tab:green', 'tab:olive', 'tab:blue', 'tab:cyan']
discharges = [c for c in list(data.columns) if '-D' in c]

scatter_matrix(data[discharges])

fig, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('Levels')
for i in range(0, len(levels)):
    ax1.plot(data.index, data[levels[i]], color=[p[1] for p in datasets if levels[i] in list(p[0].columns)][0], label=levels[i])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Discharges')
for i in range(0, len(discharges)):
    ax2.plot(data.index, data[discharges[i]], color=[p[1] for p in datasets if discharges[i] in list(p[0].columns)][0], label=discharges[i])

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

