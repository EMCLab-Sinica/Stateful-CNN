import os.path
import pprint
import statistics
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

input_data = {}
current = None
filename = sys.argv[1]
with open(filename) as f:
    for line in f:
        if line.startswith('*'):
            title = line[1:].strip()
            continue
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 1:
            input_data[parts[0]] = {}
            current = input_data[parts[0]]
            continue
        try:
            method, granularity = parts[0].split('-')
            granularity = int(granularity[1:])
        except ValueError:
            method = parts[0]
            granularity = None
        data = list(map(float, parts[1:]))
        if method not in current:
            current[method] = {'y': [], 'yerr': []}
        current[method]['y'].append(statistics.mean(data))
        current[method]['yerr'].append(statistics.stdev(data))

pprint.pprint(input_data)

matplotlib.rcParams['errorbar.capsize'] = 5

N = 3

fig, axs = plt.subplots(3)
fig.set_size_inches(6, 10)
plots = []

width = 0.25         # the width of the bars
ind = np.arange(N) + 2 * width    # the x locations for the groups
hatches = ['**', '//', r'\\', '..']
xs = np.linspace(-0.25, 3.25, 200)

for idx, model in enumerate(['LeNet/MNIST', 'SqueezeNet/CIFAR-10', 'KWS/GoogleSpeech']):
    ax = axs[idx]
    current = input_data[model]
    baseline_data = current.pop('Baseline')
    ax.bar(x=0, height=baseline_data['y'], width=width, bottom=0, fill=False, hatch=hatches[0])
    for idx, method in enumerate(['HAWAII', 'JAPARI', 'Stateful']):
        data = current[method]
        plots.append(ax.bar(x=ind + width * idx, height=data['y'], yerr=data['yerr'], width=width, bottom=0, fill=False, hatch=hatches[idx + 1]))

    ax.set_title(model)
    ax.set_xticks(np.concatenate(([0], np.arange(N) + 3 * width), axis=None))
    ax.set_xticklabels(('Baseline', '1 job', '2 jobs', '4 jobs'))

    horiz_line_data = np.array([baseline_data['y'] for i in range(len(xs))])
    ax.plot(xs, horiz_line_data, linestyle='--', color='black')

    ax.set(ylabel='Inference time (seconds)')

    ax.legend(plots, current.keys())
    ax.autoscale_view()

plt.xlabel('Progress preservation granularity')
plt.suptitle(title, y=0.95)

# plt.show()
basename = os.path.basename(filename)
name, _ = os.path.splitext(basename)
chart_path = filename.replace(basename, f'chart-{name}')
plt.savefig(chart_path + '.png')
plt.savefig(chart_path + '.pdf')
