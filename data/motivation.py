import os
import pathlib
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from generate import calculate_avg_stdev, job_desc

matplotlib.rcParams.update({
    'errorbar.capsize': 5,
    # 'font.family': 'DejaVu Serif',
    'font.size': 10,
})

def plot(dfs, device, variant, outdir):
    df = dfs[0]
    df = df[df['device'] == device]
    print(df)

    N = len(df.query('method == "HAWAII"'))

    methods = ['HAWAII', 'JAPARI']
    n_methods = len(methods)
    fig, ax = plt.subplots(1, 1)
    # width, height
    fig.set_size_inches(5, 3)
    plots = []

    width = 1.0 / (1 + n_methods)     # the width of the bars
    ind = np.arange(N) + 2 * width    # the x locations for the groups

    df = df.sort_values('batch')
    baseline_data = df.query('method == "Baseline"')
    has_baseline = variant == 'stable'
    if has_baseline:
        ax.bar(x=0, height=baseline_data['Average'], width=width, bottom=0, color='cyan', edgecolor='black')
    else:
        ax.bar(x=0, height=0, width=width, bottom=0)
        ax.annotate('×', xy=(-0.1, 5), xycoords='data', fontsize='xx-large', fontweight='bold')

    baseline_extras = dfs[1]
    # hawaii_extras = dfs[2]
    japari_extras = dfs[3]
    for idx_method, method in enumerate(methods):
        data = df[df['method'] == method]
        if data.empty:
            continue
        kwargs = {
            'x': ind + width * idx_method,
            'width': width,
            'color': 'cyan',
            'edgecolor': 'black',
        }
        if method == 'HAWAII':
            # kwargs['height'] = hawaii_extras['Average']
            kwargs['height'] = baseline_data['Average']
        elif method == 'JAPARI':
            kwargs['height'] = baseline_data['Average']
        plots.append(ax.bar(**kwargs))

        kwargs['bottom'] = list(kwargs['height'])
        kwargs['color'] = 'magenta'
        if method == 'HAWAII':
            kwargs['height'] = data['Average'].values - baseline_data['Average'].values
        elif method == 'JAPARI':
            partial_sums_cost = baseline_data['Average'].values - baseline_extras['Average'].values
            kwargs['height'] = (data['Average'].values - japari_extras['Average'].values) - partial_sums_cost
        plots.append(ax.bar(**kwargs))

        for idx, height in enumerate(data['Average'].values):
            xy = (ind[idx] + width * (idx_method - 0.4), height + 0.2)
            ax.annotate(method, xy=xy, xycoords='data', fontsize='small')

        kwargs['bottom'] = list(kwargs['height'] + kwargs['bottom'])
        kwargs['color'] = 'yellow'
        if method == 'JAPARI':
            kwargs['height'] = data['Average'].values - kwargs['bottom']
        else:
            kwargs['height'] = 0
        plots.append(ax.bar(**kwargs))

        # Put ticks at the middle bar - average of first bar (2) and the last bar (n_methods + 1)
        x_ticks = np.concatenate(([0], np.arange(N) + (2 + (n_methods + 1)) / 2 * width), axis=None)
        ax.set_xticks(x_ticks)
        x_labels = [job_desc(job) for job in df.query('method == "HAWAII"')['batch']]
        if has_baseline:
            x_labels = ['Ideal'] + x_labels
        else:
            x_labels = [''] + x_labels
        ax.set_xticklabels(x_labels)

        if has_baseline and False:
            xs = np.linspace(-0.25, N + 0.25, 200)
            horiz_line_data = np.array([baseline_data['Average'] for i in range(len(xs))])
            ax.plot(xs, horiz_line_data, linestyle='--', color='black')

        ax.set(ylabel='Processing time (seconds)')

        ax.legend(plots, ['Inference', 'Data transfer overhead', 'Computation overhead'])
        ax.set_ylim(0, 8)
        # ax.autoscale_view()

    plt.subplots_adjust(left=0.15, bottom=0.1, right=1, top=0.9, wspace=0.2)

    filename = f'chart-{device}-{variant}-motivation'
    pdf_path = outdir / (filename + '.pdf')
    plt.savefig(outdir / (filename + '.png'))
    plt.savefig(pdf_path)
    subprocess.check_call(['pdfcrop', pdf_path, outdir / (filename + '-cropped.pdf')])
    os.remove(pdf_path)

def main():
    df = pd.read_csv(pathlib.Path(__file__).parent / 'data.csv').query('config == "mnist"')
    outdir = pathlib.Path(sys.argv[1])

    motivation_data = calculate_avg_stdev(df.query('method in ["Baseline", "HAWAII", "JAPARI"]'))
    baseline_data = calculate_avg_stdev(df.query('method == "Baseline_no_output_preservation" & power == 0'))
    hawaii_data = calculate_avg_stdev(df.query('method == "HAWIII_no_footprint_preservation" & power == 0'))
    japari_data = calculate_avg_stdev(df.query('method == "JAPARI_no_output_preservation" & power == 0'))
    plot([motivation_data.query('power == 0'), baseline_data, hawaii_data, japari_data], 'msp430', 'stable', outdir)
    # plot([motivation_data.query('power == 4')], 'msp430', 'unstable', outdir)

if __name__ == '__main__':
    main()
