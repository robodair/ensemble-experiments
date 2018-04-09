import argparse

from ensemble_experiments import datagen2d, experiment2d

def main():
    """
    Console entrypoint
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    datagen_parser = subparsers.add_parser("datagen")
    datagen2d.setup_parser(datagen_parser)

    experiment2d_parser = subparsers.add_parser("ex2d")
    experiment2d.setup_parser(experiment2d_parser)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
