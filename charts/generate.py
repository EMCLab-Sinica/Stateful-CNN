import os.path
import pprint
import statistics
import sys
import textwrap

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
            method, scenario = parts[0].split('-')
        except ValueError:
            method = parts[0]
            scenario = None
        data = list(map(float, parts[1:]))
        if method not in current:
            current[method] = {'y': [], 'yerr': [], 'scenario': []}
        current[method]['y'].append(statistics.mean(data))
        current[method]['yerr'].append(statistics.stdev(data))
        current[method]['scenario'].append(scenario)

pprint.pprint(input_data)

matplotlib.rcParams['errorbar.capsize'] = 5

N = len(input_data['LeNet/MNIST']['Stateful']['y'])

fig, axs = plt.subplots(3)
fig.set_size_inches(6, 10)
plots = []

width = 0.25         # the width of the bars
ind = np.arange(N) + 2 * width    # the x locations for the groups
hatches = ['', '//', r'\\', '..']
xs = np.linspace(-0.25, 3.25, 200)
y_limits = [30, 125, 8]
y_limits = None

for idx, model in enumerate(['LeNet/MNIST', 'SqueezeNet/CIFAR-10', 'KWS/GoogleSpeech']):
    ax = axs[idx]
    if y_limits:
        ax.set_ylim(bottom=0, top=y_limits[idx])
    current = input_data[model]
    baseline_data = current.get('Baseline')
    if baseline_data:
        current.pop('Baseline')
        ax.bar(x=0, height=baseline_data['y'], width=width, bottom=0, fill=False, hatch=hatches[0])
    for idx, method in enumerate(['HAWAII', 'JAPARI', 'Stateful']):
        data = current[method]
        kwargs = {'yerr': data['yerr']} if not baseline_data else {}
        plots.append(ax.bar(x=ind + width * idx, height=data['y'], width=width, fill=False, hatch=hatches[idx + 1], **kwargs))

    ax.set_title(model)
    ax.set_xticks(np.concatenate(([0], np.arange(N) + 3 * width), axis=None))
    scenario = current['Stateful']['scenario']
    if scenario[0][0] == 'B':
        def granularity_desc(granularity):
            granularity_val = int(granularity[1:])
            unit = 'jobs' if granularity_val > 1 else 'job'
            return ' '.join((str(granularity_val), unit))
        x_labels = list(map(granularity_desc, scenario))
    else:
        x_labels = scenario
    if baseline_data:
        x_labels = ['Ideal'] + x_labels
    else:
        x_labels = [''] + x_labels
    ax.set_xticklabels(x_labels)

    if baseline_data:
        horiz_line_data = np.array([baseline_data['y'] for i in range(len(xs))])
        ax.plot(xs, horiz_line_data, linestyle='--', color='black')

    ax.set(ylabel='Inference time (seconds)')

    ax.legend(plots, current.keys())
    ax.autoscale_view()

if baseline_data:
    plt.xlabel('Progress preservation granularity')

plt.suptitle('\n'.join(textwrap.wrap(title, 50)), y=0.95)

# plt.gcf().subplots_adjust(bottom=0.25)

# plt.show()
basename = os.path.basename(filename)
name, _ = os.path.splitext(basename)
chart_path = filename.replace(basename, f'chart-{name}')
plt.savefig(chart_path + '.png')
plt.savefig(chart_path + '.pdf')
