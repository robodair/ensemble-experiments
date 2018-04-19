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
    ac_v_comp = pyplot.figure()
    ac_v_comp.suptitle("Accuracy Vs Number of Component Nets by Error Rate")
    ac_v_comp_early = ac_v_comp.add_subplot(211)
    ac_v_comp_ot = ac_v_comp.add_subplot(212)
    ac_v_comp_early.set_title("Early Exit")
    ac_v_comp_ot.set_title("Overtrained")
    # Variance difference V Components by Error Rate
    vardiff_v_comp = pyplot.figure()
    vardiff_v_comp.suptitle("Variance Difference Vs Number of Components")
    vardiff_v_comp_plot = vardiff_v_comp.add_subplot(111)
    # Varience V Components by Error Rate
    var_v_comp = pyplot.figure()
    var_v_comp.suptitle("Variance Vs Number of Components")
    var_v_comp_early = var_v_comp.add_subplot(211)
    var_v_comp_ot = var_v_comp.add_subplot(212)
    var_v_comp_early.set_title("Early Exit")
    var_v_comp_ot.set_title("Overtrained")

    marker = '.'
    # marker = '1'
    regression_degree = 10
    regression_sample = np.arange(1,99.5,0.5)
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for index, rate in enumerate(args.rates):
        color = colours[index]
        statsfile = args.save_dir / f"error-{rate}" / "anne_stats.csv"
        stats = pd.read_csv(statsfile)

        # Accuracy
        acc_regression = np.polyval(
            np.polyfit(
                stats['anne_number'],
                stats['accuracy'],
                regression_degree
            ),
            regression_sample
        )
        ot_acc_regression = np.polyval(
            np.polyfit(
                stats['anne_number'],
                stats['ot_accuracy'],
                regression_degree
            ),
            regression_sample
        )
        ac_v_comp_early.plot(
            stats['anne_number'],
            stats['accuracy'],
            marker=marker,
            label=f"Err {rate}",
            linestyle='None',
            color=color
        )
        ac_v_comp_early.plot(
            regression_sample,
            acc_regression,
            linestyle=':',
            color=color
        )
        ac_v_comp_ot.plot(
            stats['anne_number'],
            stats['ot_accuracy'],
            marker=marker,
            label=f"Err {rate}",
            linestyle='None',
            color=color
        )
        ac_v_comp_ot.plot(
            regression_sample,
            ot_acc_regression,
            linestyle=':',
            color=color,
        )

        # Variance Difference
        vardiff = stats['entropy_var'] - stats['ot_entropy_var']
        vardiff_v_comp_plot.plot(
            vardiff,
            marker=marker,
            label=f"Err {rate}",
            linestyle='None',
            color=color
        )
        vardiff_regression = np.polyval(
            np.polyfit(
                stats['anne_number'],
                vardiff,
                regression_degree
            ),
            regression_sample
        )
        vardiff_v_comp_plot.plot(
            regression_sample,
            vardiff_regression,
            linestyle=':',
            color=color,
        )

        # Variance
        var_regression = np.polyval(
            np.polyfit(
                stats['anne_number'],
                stats['entropy_var'],
                regression_degree
            ),
            regression_sample
        )
        ot_var_regression = np.polyval(
            np.polyfit(
                stats['anne_number'],
                stats['ot_entropy_var'],
                regression_degree
            ),
            regression_sample
        )
        var_v_comp_early.plot(
            stats['anne_number'],
            stats['entropy_var'],
            marker=marker,
            label=f"Err {rate}",
            linestyle='None',
            color=color
        )
        var_v_comp_early.plot(
            regression_sample,
            var_regression,
            linestyle=':',
            color=color
        )
        var_v_comp_ot.plot(
            stats['anne_number'],
            stats['ot_entropy_var'],
            marker=marker,
            label=f"Err {rate}",
            linestyle='None',
            color=color
        )
        var_v_comp_ot.plot(
            regression_sample,
            ot_var_regression,
            linestyle=':',
            color=color,
        )


    ac_v_comp_early.legend(bbox_to_anchor=(1, 0.8))
    ac_v_comp_early.grid(linestyle=':')
    ac_v_comp_ot.legend(bbox_to_anchor=(1, 0.8))
    ac_v_comp_ot.grid(linestyle=':')
    vardiff_v_comp_plot.legend(bbox_to_anchor=(1, 0.8))
    vardiff_v_comp_plot.grid(linestyle=':')
    var_v_comp_early.legend(bbox_to_anchor=(1, 0.8))
    var_v_comp_early.grid(linestyle=':')
    var_v_comp_ot.legend(bbox_to_anchor=(1, 0.8))
    var_v_comp_ot.grid(linestyle=':')

    ac_v_comp_early.set_ybound(0, 100)
    ac_v_comp_ot.set_ybound(0, 100)
    ac_v_comp_early.set_xbound(1, 99)
    ac_v_comp_ot.set_xbound(1, 99)

    var_v_comp_early.set_ybound(0, 1)
    var_v_comp_ot.set_ybound(0, 1)
    var_v_comp_early.set_xbound(1, 99)
    var_v_comp_ot.set_xbound(1, 99)

    pyplot.show(block=True) # block till window is closed


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("save_dir",
                        help="Directory to read experiment files from", type=Path)
    parser.add_argument("-r", "--rates",
                        help="Create plots for the given error rates",
                        type=list,
                        nargs="+",
                        default=[10,20,30,40, 50])
    parser.set_defaults(func=main)
