from DataImport import *
from DataApproximations import *


data_rain = import_rainfall("D:\\Наука\\_Статьи\\__в работе\\water\\data\\Eildon rainfall\\IDCJAC0009_088023_1800_Data.csv", 1)

datasets = [data_rain]

datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\Eildon head\\405258.csv", "Eildon-L", 1, 15))
#datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\GoulburnKirwans\\405282.csv", "GoulburnKirwans-L", 1, 15))
datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\GoulburnWeir\\405259.csv", "GoulburnWeir-L", 1, 18))
datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\Eildon exit\\405203.csv", "Eildon-D", 3, 18))
datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\StuartMurrey\\405700.csv", "StuartMurrey-D", 3, 15))
datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\GoulburnRiver\\405253.csv", "GoulburnRiver-D", 3, 18))
datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\EastMain\\405704.csv", "EastMain-D", 1, 25))
#datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\GoulburnMurchison\\405200.csv", "GoulburnMurchison-D", 5, 25))
datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\CattanachWaranga\\405702.csv", "Cattanach-D", 3, 25))

colors = {
    'Eildon-L': 'tab:orange',
    'GoulburnWeir-L': 'tab:red',
    'Eildon-D': 'tab:cyan',
    'StuartMurrey-D': 'tab:green',
    'GoulburnRiver-D': 'tab:olive',
    'EastMain-D': 'tab:gray',
    'Cattanach-D': 'tab:blue'
}


data = pd.concat(datasets, axis=1)

#data.drop(data[data.index < pd.Timestamp(2018, 10, 1)].index, inplace=True)
#data.drop(data[data.index < pd.Timestamp(2001, 1, 1)].index, inplace=True)


data_year_dropped = data.copy()

data_year_dropped.index = data_year_dropped.index.map(lambda t: t.date() if type(t) is pd.Timestamp else t)
data_year_dropped.index = data_year_dropped.index.map(lambda t: t.replace(year=2020))
data_year_dropped = data_year_dropped.groupby(level=0).mean()
data_year_dropped.drop(data[data.index == pd.Timestamp(2020, 2, 29)].index, inplace=True)
data_year_dropped.index = data_year_dropped.index.map(lambda t: (t.timetuple().tm_yday - 1) / 365.0)




levels = [c for c in list(data.columns) if '-L' in c]

discharges = [c for c in list(data.columns) if '-D' in c]

# plot year averages
# fig, ax1 = plt.subplots()
# ax1.set_xlabel('time')
# ax1.set_ylabel('Levels')
# for i in range(0, len(levels)):
#     ax1.scatter(data_year_dropped.index, data_year_dropped[levels[i]], color=colors[levels[i]], label=levels[i])
#
# ax2 = ax1.twinx()
#
# ax2.set_ylabel('Discharges')
# for i in range(0, len(discharges)):
#     ax2.scatter(data_year_dropped.index, data_year_dropped[discharges[i]], color=colors[discharges[i]], label=discharges[i])
#
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
#
# fig.tight_layout()
# plt.show()
#


approximations = []
approximations.append(approximation('Rainfall', data_year_dropped.index, data_year_dropped['rainfall'], 14, 2, 5))

# get missing Goulburn data
data_year_dropped['GoulburnRiver-D'] = data_year_dropped['Eildon-D'] - data_year_dropped[['StuartMurrey-D', 'EastMain-D', 'Cattanach-D']].sum(axis=1)


discharges_approx = ['Eildon-D', 'StuartMurrey-D', 'EastMain-D', 'Cattanach-D', 'GoulburnRiver-D']

for s in discharges_approx:
    approximations.append(approximation(s, data_year_dropped.index, data_year_dropped[s], 3, 40, 40))

for a in approximations:
    a.plot_all()
    a.plot_rmses()

