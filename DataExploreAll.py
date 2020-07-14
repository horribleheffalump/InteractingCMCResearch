from pandas.plotting import scatter_matrix
import seaborn as sns


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

data.drop(data[data.index < pd.Timestamp(2018, 10, 1)].index, inplace=True)
#data.drop(data[data.index < pd.Timestamp(2001, 1, 1)].index, inplace=True)


data['GoulburnOutput'] = data[['StuartMurrey-D','GoulburnRiver-D','EastMain-D','Cattanach-D']].dropna().sum(axis =1)
data['GoulburnOutput3ch'] = data[['StuartMurrey-D','EastMain-D','Cattanach-D']].dropna().sum(axis =1)


c = data.corr()
sns.heatmap(c, xticklabels=c.columns, yticklabels=c.columns, cmap='coolwarm')
c.style.background_gradient(cmap='coolwarm')
plt.matshow(c)
plt.plot()

#g = sns.pairplot(data, palette="YlGnBu")
#g.savefig("pairplot.png")



levels = [c for c in list(data.columns) if '-L' in c]

discharges = [c for c in list(data.columns) if '-D' in c]

# pairs
scatter_matrix(data[discharges])
scatter_matrix(data[["Eildon-D", 'GoulburnOutput', 'GoulburnOutput3ch']])

# plot all data
fig, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('Levels')
for i in range(0, len(levels)):
    ax1.plot(data.index, data[levels[i]], color=colors[levels[i]], label=levels[i])

ax2 = ax1.twinx()

ax2.set_ylabel('Discharges')
for i in range(0, len(discharges)):
    ax2.plot(data.index, data[discharges[i]], color=colors[discharges[i]], label=discharges[i])
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.show()

