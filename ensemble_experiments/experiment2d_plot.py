"""
Create plots for experiment2d, must be run after experiment 2d has completed
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np

def main(args):
    pyplot.ion()

    # Accuracy Vs Num Components by Error Rate
    ac_v_comp_fig = pyplot.figure()
    ac_v_comp_fig.suptitle("Accuracy Vs Number of Component Nets by Error Rate")
    ac_v_comp = ac_v_comp_fig.add_subplot(111)
    # Variance difference V Components by Error Rate
    vardiff_v_comp = pyplot.figure()
    vardiff_v_comp.suptitle("Variance Difference (Early Exit - Overtrained) Vs Number of Components")
    vardiff_v_comp_plot = vardiff_v_comp.add_subplot(111)
    # Varience V Components by Error Rate
    var_v_comp_fig = pyplot.figure()
    var_v_comp_fig.suptitle("Variance Vs Number of Components")
    var_v_comp = var_v_comp_fig.add_subplot(111)

    marker = '.'
    # marker = '1'
    regression_degree = 5
    regression_sample = np.arange(args.lower_comp,args.upper_comp+0.5,0.5)
    colours = [('xkcd:light blue', 'xkcd:blue'), ('xkcd:light green', 'xkcd:green'),
               ('xkcd:salmon pink', 'xkcd:deep pink'), ('xkcd:light purple', 'xkcd:purple'),
               ('xkcd:grey', 'xkcd:dark grey')]
    for index, rate in enumerate([10,20,30,40, 50]):
        if rate not in args.rates:
            continue
        color, color_ot = colours[index]
        statsfile = args.save_dir / f"error-{rate}" / "anne_stats.csv"
        stats = pd.read_csv(statsfile)

        anne_number = stats['anne_number'][args.lower_comp-1:args.upper_comp-1:args.step]
        accuracy = stats['accuracy'][args.lower_comp-1:args.upper_comp-1:args.step]
        ot_accuracy = stats['ot_accuracy'][args.lower_comp-1:args.upper_comp-1:args.step]
        entropy_var = stats['entropy_var'][args.lower_comp-1:args.upper_comp-1:args.step]
        ot_entropy_var = stats['ot_entropy_var'][args.lower_comp-1:args.upper_comp-1:args.step]
        # Accuracy
        acc_regression = np.polyval(
            np.polyfit(
                anne_number,
                accuracy,
                regression_degree
            ),
            regression_sample
        )
        ot_acc_regression = np.polyval(
            np.polyfit(
                anne_number,
                ot_accuracy,
                regression_degree
            ),
            regression_sample
        )
        ac_v_comp.plot(
            anne_number,
            accuracy,
            marker=marker,
            label=f"Err {rate}",
            linestyle='None',
            color=color
        )
        ac_v_comp.plot(
            regression_sample,
            acc_regression,
            linestyle=':',
            color=color
        )
        ac_v_comp.plot(
            anne_number,
            ot_accuracy,
            marker=marker,
            label=f"OT Err {rate}",
            linestyle='None',
            color=color_ot
        )
        ac_v_comp.plot(
            regression_sample,
            ot_acc_regression,
            linestyle=':',
            color=color_ot,
        )

        # Variance Difference
        vardiff = entropy_var - ot_entropy_var
        vardiff_v_comp_plot.plot(
            vardiff,
            marker=marker,
            label=f"Err {rate}",
            linestyle='None',
            color=color_ot
        )
        vardiff_regression = np.polyval(
            np.polyfit(
                anne_number,
                vardiff,
                regression_degree
            ),
            regression_sample
        )
        vardiff_v_comp_plot.plot(
            regression_sample,
            vardiff_regression,
            linestyle=':',
            color=color_ot,
        )

        # Variance
        var_regression = np.polyval(
            np.polyfit(
                anne_number,
                entropy_var,
                regression_degree
            ),
            regression_sample
        )
        ot_var_regression = np.polyval(
            np.polyfit(
                anne_number,
                ot_entropy_var,
                regression_degree
            ),
            regression_sample
        )
        var_v_comp.plot(
            anne_number,
            entropy_var,
            marker=marker,
            label=f"Err {rate}",
            linestyle='None',
            color=color
        )
        var_v_comp.plot(
            regression_sample,
            var_regression,
            linestyle=':',
            color=color
        )
        var_v_comp.plot(
            anne_number,
            ot_entropy_var,
            marker=marker,
            label=f"OT Err {rate}",
            linestyle='None',
            color=color_ot
        )
        var_v_comp.plot(
            regression_sample,
            ot_var_regression,
            linestyle=':',
            color=color_ot,
        )


    ac_v_comp.legend(bbox_to_anchor=(1, 0.8))
    ac_v_comp.grid(linestyle=':')
    ac_v_comp.set_xlabel("Number of Component Networks")
    ac_v_comp.set_ylabel("Accuracy")
    vardiff_v_comp_plot.legend(bbox_to_anchor=(1, 0.8))
    vardiff_v_comp_plot.grid(linestyle=':')
    vardiff_v_comp_plot.set_xlabel("Number of Component Networks")
    vardiff_v_comp_plot.set_ylabel("Variance Difference (OverTrained - EarlyExit)")
    var_v_comp.legend(bbox_to_anchor=(1, 0.8))
    var_v_comp.grid(linestyle=':')
    var_v_comp.set_xlabel("Number of Component Networks")
    var_v_comp.set_ylabel("Entropy Variance Measure")

    if args.fix_axes:
        ac_v_comp.set_ybound(20, 100)
        ac_v_comp.set_xbound(1, 99)

        var_v_comp.set_ybound(0, 0.7)
        var_v_comp.set_xbound(1, 99)

    pyplot.show(block=True) # block till window is closed


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("save_dir",
                        help="Directory to read experiment files from", type=Path)
    parser.add_argument("-r", "--rates",
                        help="Create plots for the given error rates",
                        type=int,
                        nargs="+",
                        default=[10,20,30,40, 50])
    parser.add_argument("-u", "--upper_comp",
                        help="Upper number of component nets to graph to",
                        type=int, default=99)
    parser.add_argument("-l", "--lower_comp",
                        help="Lower number of component nets to graph to",
                        type=int, default=5)
    parser.add_argument("-s", "--step",
                        help="Step to use when slicing stats data (e.g. only plot every 2nd)",
                        type=int, default=1)
    parser.add_argument("-x", "--fix-axes", action='store_true',
                        help="Fix axes sizes to sensible defaults, useful for plot comparison")

    parser.set_defaults(func=main)
