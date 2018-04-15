import argparse

import matplotlib
matplotlib.use('TKAgg')

from ensemble_experiments import dataviz, experiment2d, trainviz

def main():
    """
    Console entrypoint
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    dataviz_parser = subparsers.add_parser("dataviz")
    dataviz.setup_parser(dataviz_parser)

    experiment2d_parser = subparsers.add_parser("ex2d")
    experiment2d.setup_parser(experiment2d_parser)

    trainviz_parser = subparsers.add_parser("trainviz")
    trainviz.setup_parser(trainviz_parser)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
