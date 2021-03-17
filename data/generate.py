import enum
import pathlib
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams.update({
    'errorbar.capsize': 5,
    # 'font.family': 'DejaVu Serif',
    'font.size': 12,
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

class XAxisType(enum.Enum):
    GRANULARITY = enum.auto()
    POWER = enum.auto()

def plot(df, device, variant, outdir):
    df = df[df['device'] == device]
    print(df)

    N = len(df.query(f'config == "{MOTIVATION_CONFIG}" & method == "HAWAII"'))

    config_names = {
        'mnist': 'CNN/MNIST',
        'cifar10': 'SqueezeNet/CIFAR-10',
        'kws': 'KWS/Speech Commands',
    }

    ylim = {
        'msp430': [150, 500, 15],
        'msp432': [128, 320, 16],
    }
    yticks = {
        'msp430': [np.arange(0, 121, step=30), np.arange(0, 401, step=100), np.arange(0, 13, step=3)],
        'msp432': [np.arange(0, 121, step=40), np.arange(0, 301, step=100), np.arange(0, 16, step=5)],
    }

    n_configs = df['config'].nunique()
    n_methods = df.query('method != "Baseline"')['method'].nunique()
    n_batches = df['batch'].nunique()
    fig, axs = plt.subplots(1, n_configs)
    if n_configs == 1:
        # width, height
        fig.set_size_inches(5, 3)
    elif n_configs == 3:
        fig.set_size_inches(11, 4)
    else:
        raise ValueError('Unexpected number of configurations!')
    plots = []

    width = 1.0 / (1 + n_methods)     # the width of the bars
    ind = np.arange(N) + 2 * width    # the x locations for the groups
    hatches = ['', '--', r'\\', 'xx']
    xs = np.linspace(-0.25, N + 0.25, 200)

    def job_desc(job):
        if job == 1:
            return '1 job'
        return f'{job} jobs'

    methods = ['HAWAII', 'JAPARI', 'Stateful']

    if variant == 'unstable' and n_configs > 1:
        x_axis = XAxisType.POWER
    else:
        x_axis = XAxisType.GRANULARITY

    for idx_config, config in enumerate(config_names.keys()):
        if n_configs == 1:
            ax = axs
        else:
            ax = axs[idx_config]
        current = df[df['config'] == config]
        if current.empty:
            continue

        if n_batches == 2:
            p = current.query('method == "Baseline"')['Average'].iat[0] or 10
            ax.set_ylim([0, p * 5.5])
        elif n_configs > 1:
            ax.set_ylim([0, ylim[device][idx_config]])
            ax.set_yticks(yticks[device][idx_config])

        if x_axis == XAxisType.POWER:
            current = current.sort_values('power')
        elif x_axis == XAxisType.GRANULARITY:
            current = current.sort_values('batch')
        baseline_data = current.query('method == "Baseline"')
        has_baseline = variant == 'stable'
        if has_baseline:
            ax.bar(x=0, height=baseline_data['Average'], width=width, bottom=0, fill=False, hatch=hatches[0])
        elif n_configs == 1:
            ax.bar(x=0, height=0, width=width, bottom=0, fill=False, hatch=hatches[0])
            ax.annotate('×', xy=(-0.1, 5), xycoords='data', fontsize='xx-large', fontweight='bold')
        stateful_time = current.query('method == "Stateful"')['Average']
        max_height = current.max()['Average']
        for idx_method, method in enumerate(methods):
            data = current[current['method'] == method]
            if data.empty:
                continue
            kwargs = {
                'x': ind + width * idx_method,
                'height': data['Average'],
                'width': width,
                'fill': False,
                'hatch': hatches[idx_method + 1],
            }
            if not has_baseline:
                kwargs['yerr'] = data['Stdev']
            plots.append(ax.bar(**kwargs))

            # Based on a function from Daniel Tsai
            if x_axis == XAxisType.POWER:
                for value_idx, height in enumerate(data['Average']):
                    if not data.query('method == "Stateful"').empty:
                        continue
                    yerr = data['Stdev'].iat[value_idx]
                    xy = [value_idx + 0.38 + width * idx_method, height + max_height * 0.03 + yerr]
                    if not height:
                        xy[0] += 0.04
                        ax.annotate('×', xy=xy, xycoords='data', fontsize='xx-large', fontweight='bold')
                        continue
                    ax.annotate('-{}%'.format(int((1 - stateful_time.iat[value_idx] / height) * 100)),
                                xy=xy, xycoords='data', annotation_clip=False, fontsize='small')

        if n_configs > 1:
            ax.set_title(config_names[config])
        # Put ticks at the middle bar - average of first bar (2) and the last bar (n_methods + 1)
        x_ticks = np.concatenate(([0], np.arange(N) + (2 + (n_methods + 1)) / 2 * width), axis=None)
        ax.set_xticks(x_ticks)
        if x_axis == XAxisType.POWER:
            x_labels = [f'{power}mW' for power in current.query('method == "HAWAII"')['power']]
        elif x_axis == XAxisType.GRANULARITY:
            x_labels = [job_desc(job) for job in current.query('method == "HAWAII"')['batch']]
        if has_baseline or n_configs == 1:
            x_labels = ['Ideal'] + x_labels
        else:
            x_labels = [''] + x_labels
        ax.set_xticklabels(x_labels)

        if has_baseline:
            horiz_line_data = np.array([baseline_data['Average'] for i in range(len(xs))])
            ax.plot(xs, horiz_line_data, linestyle='--', color='black')

        if idx_config == 0:
            ax.set(ylabel='Inference time (seconds)')

        if n_configs == 1:
            ax.legend(plots, methods)
        ax.autoscale_view()

    if n_configs > 1:
        plt.figlegend(plots, methods, ncol=len(methods))

    plt.subplots_adjust(left=0.15, bottom=0.1, right=1, top=0.8, wspace=0.2)

    filename = f'{device}-{variant}'
    if n_configs == 1 and n_batches != 2:
        filename += '-motivation'
    filename = 'chart-' + filename
    plt.savefig(outdir / (filename + '.png'))
    plt.savefig(outdir / (filename + '.pdf'))
    subprocess.check_call(['pdfcrop', outdir / (filename + '.pdf'), outdir / (filename + '-cropped.pdf')])

def main():
    df = pd.read_csv(pathlib.Path(__file__).parent / 'data.csv')
    outdir = pathlib.Path(sys.argv[1])

    motivation_data = calculate_avg_stdev(df.query(f'config == "{MOTIVATION_CONFIG}" & method != "Stateful"'))
    plot(motivation_data.query('power == 0'), 'msp430', 'stable', outdir)
    plot(motivation_data.query('power == 4'), 'msp430', 'unstable', outdir)

    for device in ['msp430', 'msp432']:
        unstable_data = calculate_avg_stdev(df.query('power > 0 & batch == 1'))
        plot(unstable_data, device, 'unstable', outdir)

        stable_data = calculate_avg_stdev(df.query('(power == 0 & batch == 1) | method == "Baseline"'))
        plot(stable_data, device, 'stable', outdir)

if __name__ == '__main__':
    main()
