from DELWPdata.DataImport import *
from DELWPdata.DataApproximations import *
from matplotlib import gridspec
from matplotlib import rc

rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('text.latex',unicode=True)

#recalc = False

fig_path = 'D:\\projects.git\\InteractingCMCResearch\\output\\'
filename_template = "D:\\Наука\\_Статьи\\__в работе\\water\\data\\_discharges_approximations\\[param]_approx.npy"
discharges_approx = ['Eildon-D', 'StuartMurrey-D', 'Cattanach-D', 'EastMain-D', 'GoulburnRiver-D']
all_series = discharges_approx + ['Rainfall']
points = np.arange(0, 1.0, 1.0 / 365)

data_rain = import_rainfall("D:\\Наука\\_Статьи\\__в работе\\water\\data\\Eildon rainfall\\IDCJAC0009_088023_1800_Data.csv", 1)

datasets = [data_rain]

datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\Eildon exit\\405203.csv", "Eildon-D", 3, 18))

datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\StuartMurrey\\405700.csv", "StuartMurrey-D", 3, 15))
datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\EastMain\\405704.csv", "EastMain-D", 1, 25))
datasets.append(import_data("D:\\Наука\\_Статьи\\__в работе\\water\\data\\CattanachWaranga\\405702.csv", "Cattanach-D", 3, 25))

colors = {
    'Eildon-D': 'tab:cyan',
    'StuartMurrey-D': 'tab:green',
    'GoulburnRiver-D': 'tab:olive',
    'EastMain-D': 'tab:gray',
    'Cattanach-D': 'tab:blue'
}


data = pd.concat(datasets, axis=1)

data_year_dropped = data.copy()

data_year_dropped.index = data_year_dropped.index.map(lambda t: t.date() if type(t) is pd.Timestamp else t)
data_year_dropped.index = data_year_dropped.index.map(lambda t: t.replace(year=2020))
data_year_dropped = data_year_dropped.groupby(level=0).mean()
data_year_dropped.drop(data[data.index == pd.Timestamp(2020, 2, 29)].index, inplace=True)
data_year_dropped.index = data_year_dropped.index.map(lambda t: (t.timetuple().tm_yday - 1) / 365.0)

# get missing Goulburn data
data_year_dropped['GoulburnRiver-D'] = data_year_dropped['Eildon-D'] - data_year_dropped[['StuartMurrey-D', 'EastMain-D', 'Cattanach-D']].sum(axis=1)

# set minimum discharge
min_d = 500
for s in discharges_approx:
    data_year_dropped[s][data_year_dropped[s] < min_d] = min_d

#if recalc:
    # approximations
approximations = {}
approximations.update({'Rainfall' : approximation('Rainfall', data_year_dropped.index.values, data_year_dropped['Rainfall'], 14, 4, 4)})
for s in discharges_approx:
    approximations.update({s: approximation(s, data_year_dropped.index.values, data_year_dropped[s], 3, 10, 10)})

total_int = 0.0
for s in ['GoulburnRiver-D', 'StuartMurrey-D', 'EastMain-D', 'Cattanach-D']:
    total_int += np.sum(approx(approximations[s].coeffs[0], points))
rainfall_int = np.sum(approx(approximations['Rainfall'].coeffs[0], points))

approximations['Rainfall'].coeffs[0] = total_int / rainfall_int * approximations['Rainfall'].coeffs[0]

for s in all_series:
    np.save(filename_template.replace('[param]', approximations[s].label), approximations[s].coeffs[0])

#else:
all_params = {}
for s in all_series:
    all_params.update({s : np.load(filename_template.replace('[param]', s))})


x_ticks = np.arange(0.5,1.5, 1.0/12) + 1.0/24
x_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

plot_points =  np.arange(0.5, 1.5, 0.001)
fig = plt.figure(figsize=(10, 3), dpi=150)
#gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[4, 1])
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

ax = plt.subplot(gs[0, 0])
ax.plot(plot_points, approx(all_params['Rainfall'], plot_points), label=s, color='red')
ax.scatter(data_year_dropped.index.values + 0.5,
           total_int / rainfall_int * np.roll(data_year_dropped[s].values, int(data_year_dropped[s].size / 2)),
               marker='.', color='red', s=1)

ax.set_ylabel('Rainfall water income [Ml/day]')

#ax.get_xaxis().set_visible(False)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)
#ax1.legend(loc='upper center')

ax = plt.subplot(gs[0, 1]) # legend
ax.plot([], [], label='Rainfall', color='red')
ax.set_axis_off()
ax.legend(loc='upper left')



fig.tight_layout()

#plt.show()
filename = f'{fig_path}rainfall.pdf'
plt.tight_layout()
plt.savefig(filename)
plt.close(fig)


fig = plt.figure(figsize=(10, 3), dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
#ax = plt.subplot(gs[1, 0])
ax = plt.subplot(gs[0, 0])
for s in discharges_approx:
    ax.plot(plot_points, approx(all_params[s], plot_points), label=s)
    ax.scatter(data_year_dropped.index.values + 0.5,
               np.roll(data_year_dropped[s].values, int(data_year_dropped[s].size / 2)), marker='.', s=1)

ax.set_ylabel('Discharges [Ml/day]')


ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)

#ax1.legend(loc='upper center')

#ax = plt.subplot(gs[1, 1]) # legend
ax = plt.subplot(gs[0, 1]) # legend
for s, correct_name in (zip(discharges_approx, ['Goulburn higher', 'Stuart Murrey', 'Cattanach', 'East Main', 'Goulburn lower'])):
    ax.plot([], [], label=correct_name)
ax.set_axis_off()
ax.legend(loc='lower left')


fig.tight_layout()

#plt.show()
filename = f'{fig_path}discharges.pdf'
plt.tight_layout()
plt.savefig(filename)
plt.close(fig)
