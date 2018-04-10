"""
Glue code for conducting experiment 1
"""
import os
# os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
# No verbose logging from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


import argparse
import signal
import numpy as np
from pathlib import Path
import keras
import pandas


def train(train, test, save_dir, epoch_max, verbose):
    save_net = save_dir / "net.h5"
    if not save_net.exists():
        import ensemble_experiments.autotrain as at
        net = at.autotrain(train, test, min_nodes=1, max_nodes=9, epoch_max=epoch_max, verbose=verbose)
        print(f"    SAVING {save_net}")
        net.save(save_net)
        return save_net
    else:
        print(f"    EXISTING {save_net}")
        return save_net

def main(args):
    print("Running Experiment")
    np.random.seed(2000)
    import ensemble_experiments.datagen2d as dg

    exdir = Path(f"{args.prefix}experiment-EPOCH_MAX={args.epoch_max}-DATA_SIZE={args.data_size}")
    exdir.mkdir(exist_ok=True)

    print(f"BEGIN RUN FOR ERROR RATE {args.error_rate}%")
    ratedir = exdir / f"error-{args.error_rate}"
    ratedir.mkdir(exist_ok=True)

    data = dg.generate_data(args.data_size, args.error_rate, 2017)
    train_data = data[::2]
    test_data = data[1::2]

    val_data = dg.generate_data(1000, args.error_rate, 2018)

    train_data.to_csv(ratedir / "initial_train.csv")
    test_data.to_csv(ratedir / "initial_test.csv")
    val_data.to_csv(ratedir / "validation.csv")

    for num_component_nets in range(1, args.num_nets+1):
        print(f" BEGIN RUN FOR {num_component_nets} NETS")
        working_dir = ratedir / f"ANNE-{num_component_nets}"
        working_dir.mkdir(exist_ok=True)
        train_test_data = []
        # Generate num_component_nets bags of data and train networks
        for i in range(num_component_nets):
            save_dir = working_dir / f"net-{i}"
            save_train = save_dir / "train.csv"
            save_test = save_dir / "test.csv"
            save_dir.mkdir(exist_ok=True)
            if not save_train.exists() and not save_test.exists():
                train_bag = train_data.sample(len(train_data), replace=True)
                test_bag = test_data.sample(len(test_data), replace=True)
            else:
                train_bag = pandas.read_csv(save_train)
                test_bag = pandas.read_csv(save_test)

            train_test_data.append((
                train_bag, test_bag, save_dir
            ))
            train_bag.to_csv(save_train)
            test_bag.to_csv(save_test)

        from multiprocessing.pool import Pool

        def wrap_train(x):
            return train(*x)

        arguments = [(*x, args.epoch_max, args.verbose) for x in train_test_data]
        networks = list(map(wrap_train, arguments))

        # TODO: Run validation across component networks and aggregate results into predictions, compare with validation (real) classes
        print(f"Nets for ANNE {num_component_nets}: {networks}")

def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--verbose", action='store_true', help="Show training logs verbosely")
    parser.add_argument("--epoch-max", type=int, help="Maximum epochs to test to", default=1500)
    parser.add_argument("--error-rate", type=int, help="Error rate to use", default=10)
    parser.add_argument("--data-size", type=int, help="Size of train/test dataset", default=300)
    parser.add_argument("--prefix", help="Prefix for the experiment folder", default="")
    parser.add_argument("--num-nets", type=int, help="Number of component nets to do the experiment to", default=99)
    parser.set_defaults(func=main)
