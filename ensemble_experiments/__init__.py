import argparse

import matplotlib
matplotlib.use('TKAgg')

from ensemble_experiments import dataviz, experiment2d, trainviz, experiment2d_plot

def main():
    """
    Console entrypoint
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    dataviz_parser = subparsers.add_parser("dataviz", help="Visualise data distribution")
    dataviz.setup_parser(dataviz_parser)

    experiment2d_parser = subparsers.add_parser("ex2d", help="Run Training & Produce ANNE Stats")
    experiment2d.setup_parser(experiment2d_parser)

    experiment2d_plot_parser = subparsers.add_parser("ex2dplot", help="Create plots for an Experiment")
    experiment2d_plot.setup_parser(experiment2d_plot_parser)

    trainviz_parser = subparsers.add_parser("trainviz", help="Visualise network training")
    trainviz.setup_parser(trainviz_parser)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
