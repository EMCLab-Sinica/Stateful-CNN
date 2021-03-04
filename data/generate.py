import pathlib
import textwrap
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams.update({
    'errorbar.capsize': 5,
    'font.size': 14,
})

MOTIVATION_CONFIG = 'mnist'

def calculate_avg_stdev(df):
    timings = df.loc[:, '0':'9']
    average = timings.mean(axis=1)
    stdev = timings.std(axis=1)
    df = df.drop(columns=[str(idx) for idx in range(10)])
    df['Average'] = average
    df['Stdev'] = stdev
    return df

def plot(df, device, variant, outdir):
    df = df[df['device'] == device]
    print(df)

    N = len(df.query(f'config == "{MOTIVATION_CONFIG}" & method == "HAWAII"'))

    config_names = {
        'mnist': 'LeNet/MNIST',
        'cifar10': 'SqueezeNet/CIFAR-10',
        'kws': 'KWS/Google Speech',
    }

    n_configs = df['config'].nunique()
    n_methods = df.query('method != "Baseline"')['method'].nunique()
    fig, axs = plt.subplots(n_configs)
    if n_configs == 1:
        plot_height = 3
    elif n_configs == 3:
        plot_height = 10
    else:
        raise ValueError('Unexpected number of configurations!')
    fig.set_size_inches(6, plot_height)
    plots = []

    width = 1.0 / (1 + n_methods)     # the width of the bars
    ind = np.arange(N) + 2 * width    # the x locations for the groups
    hatches = ['', 'o', r'\\', '..']
    xs = np.linspace(-0.25, 3.25, 200)

    titles = {
        'stable': 'Stable power',
        'unstable': 'Intermittent execution (100uF)',
    }

    def job_desc(job):
        if job == 1:
            return '1 job'
        return f'{job} jobs'

    methods = ['HAWAII', 'JAPARI', 'Stateful']

    x_axis_use_power = variant == 'unstable' and n_configs > 1

    for idx, config in enumerate(config_names.keys()):
        if n_configs == 1:
            ax = axs
        else:
            ax = axs[idx]
        current = df[df['config'] == config]
        if current.empty:
            continue

        ax.set_ylim([0, current['Average'].max() * 1.3])

        if x_axis_use_power:
            current = current.sort_values('power')
        else:
            current = current.sort_values('batch')
        baseline_data = current.query('method == "Baseline"')
        has_baseline = variant == 'stable'
        if has_baseline:
            ax.bar(x=0, height=baseline_data['Average'], width=width, bottom=0, fill=False, hatch=hatches[0])
        elif n_configs == 1:
            ax.bar(x=0, height=0, width=width, bottom=0, fill=False, hatch=hatches[0])
        stateful_time = current.query('method == "Stateful"')['Average']
        max_height = current.max()['Average']
        for idx, method in enumerate(methods):
            data = current[current['method'] == method]
            if data.empty:
                continue
            kwargs = {
                'x': ind + width * idx,
                'height': data['Average'],
                'width': width,
                'fill': False,
                'hatch': hatches[idx + 1],
            }
            if not has_baseline:
                kwargs['yerr'] = data['Stdev']
            plots.append(ax.bar(**kwargs))

            # Based on a function from Daniel Tsai
            if x_axis_use_power:
                for value_idx, height in enumerate(data['Average']):
                    if not data.query('method == "Stateful"').empty or not height:
                        continue
                    yerr = list(data['Stdev'])[value_idx]
                    xy = (value_idx + 0.4 + width * idx, height + max_height * 0.04 + yerr)
                    ax.annotate('-{}%'.format(int((1 - list(stateful_time)[value_idx] / height) * 100)),
                                xy=xy, xycoords='data', annotation_clip=False)

        if n_configs > 1:
            ax.set_title(config_names[config])
        # Put ticks at the middle bar - average of first bar (2) and the last bar (n_methods + 1)
        ax.set_xticks(np.concatenate(([0], np.arange(N) + (2 + (n_methods + 1)) / 2 * width), axis=None))
        if x_axis_use_power:
            x_labels = [f'{power}mW' for power in current.query('method == "HAWAII"')['power']]
        else:
            x_labels = [job_desc(job) for job in current.query('method == "HAWAII"')['batch']]
        if has_baseline or n_configs == 1:
            x_labels = ['Ideal'] + x_labels
        else:
            x_labels = [''] + x_labels
        ax.set_xticklabels(x_labels)

        if has_baseline:
            horiz_line_data = np.array([baseline_data['Average'] for i in range(len(xs))])
            ax.plot(xs, horiz_line_data, linestyle='--', color='black')

        ax.set(ylabel='Inference time (seconds)')

        ax.legend(plots, methods, fontsize='small')
        ax.autoscale_view()

    if not x_axis_use_power:
        plt.xlabel('Progress preservation granularity')

    title = titles.get(variant)
    if title:
        plt.suptitle('\n'.join(textwrap.wrap(title, 50)), y=0.98)

    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.9, hspace=0.5)

    filename = f'{device}-{variant}'
    if n_configs == 1:
        filename += '-motivation'
    filename = 'chart-' + filename
    plt.savefig(outdir / (filename + '.png'))
    plt.savefig(outdir / (filename + '.pdf'))

def main():
    df = pd.read_csv(pathlib.Path(__file__).parent / 'data.csv')
    outdir = pathlib.Path(sys.argv[1])

    motivation_data = calculate_avg_stdev(df.query(f'config == "{MOTIVATION_CONFIG}" & method != "Stateful"'))
    plot(motivation_data.query('power == 0'), 'msp430', 'stable', outdir)
    plot(motivation_data.query('power == 4'), 'msp430', 'unstable', outdir)

    for device in ['msp430', 'msp432']:
        unstable_data = calculate_avg_stdev(df.query('power > 0 & batch == 1'))
        plot(unstable_data, device, 'unstable', outdir)

        stable_data = calculate_avg_stdev(df.query('power == 0'))
        plot(stable_data, device, 'stable', outdir)

if __name__ == '__main__':
    main()
