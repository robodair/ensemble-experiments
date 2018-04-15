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
import pandas


def train(train_path, test_path, save_dir, epoch_max, verbose):
    save_net = save_dir / "net.h5"
    save_overtrained_net = save_dir / "overtrained_net.h5"
    if verbose:
        print(f" Training for {save_net}")
    if not save_net.exists() or not save_overtrained_net.exists():
        train = pandas.read_csv(train_path)
        test = pandas.read_csv(test_path)
        import ensemble_experiments.autotrain as at
        net, overtrained_net = at.autotrain(train, test, min_nodes=1, max_nodes=9,
            max_epochs=epoch_max, verbose=verbose)
        print(f"    SAVING {save_net}")
        net.save(save_net)
        overtrained_net.save(save_overtrained_net)
        return save_net, save_overtrained_net
    else:
        print(f"    EXISTING {save_net}")
        return save_net, save_overtrained_net

def wrap_train(x):
    return train(*x)

def main(args):
    print("Running Experiment")
    np.random.seed(2000)
    import keras
    import ensemble_experiments.datagen2d as dg

    exdir = Path(f"{args.prefix}experiment-EPOCH_MAX={args.epoch_max}-DATA_SIZE={args.data_size}")
    exdir.mkdir(exist_ok=True)

    print(f"BEGIN RUN FOR ERROR RATE {args.error_rate}%")
    ratedir = exdir / f"error-{args.error_rate}"
    ratedir.mkdir(exist_ok=True)

    data_csv = ratedir / "data.csv"
    if not data_csv.exists():
        data = dg.generate_data(args.data_size, args.error_rate, 2017)
        data.to_csv(data_csv)
    else:
        data = pandas.read_csv(data_csv)

    val_data_csv = ratedir / "validation.csv"
    if not val_data_csv.exists():
        val_data = dg.generate_data(1000, args.error_rate, 2018)
        val_data.to_csv(val_data_csv)
    else:
        val_data = pandas.read_csv(val_data_csv)


    train_data = data[::2]
    test_data = data[1::2]

    all_arguments = []

    for num_component_nets in range(1, args.num_nets+1):
        print(f" GENERATE RUN DATA FOR {num_component_nets} NETS")
        working_dir = ratedir / f"ANNE-{num_component_nets}"
        working_dir.mkdir(exist_ok=True)
        train_test_data = []
        # Generate num_component_nets bags of data and train networks
        for i in range(num_component_nets):
            save_dir = working_dir / f"net-{i}"
            save_train = save_dir / "train.csv"
            save_test = save_dir / "test.csv"
            save_dir.mkdir(exist_ok=True)
            if not save_train.exists():
                train_bag = train_data.sample(len(train_data), replace=True)
                train_bag.to_csv(save_train)
            if not save_test.exists():
                test_bag = test_data.sample(len(test_data), replace=True)
                test_bag.to_csv(save_test)
            train_test_data.append((
                save_train, save_test, save_dir
            ))

        all_arguments += [(*x, args.epoch_max, args.verbose) for x in train_test_data]

    from multiprocessing.pool import Pool
    p = Pool(processes=8)
    network_paths = p.map(wrap_train, all_arguments)

    # TODO: Run validation across component networks and aggregate results into predictions
    # compare with validation (real) classes
    print(f"Nets for ANNE: {network_paths}")

    val_data_input = val_data.as_matrix(columns=('x', 'y'))
    val_data_classes = val_data["realclass"].values
    for net_path, overtrained_path in network_paths:
        op_model = keras.models.load_model(net_path)
        res = op_model.evaluate(
            val_data_input,
            val_data_classes,
            verbose=1
        )
        print(op_model.metrics_names)
        print(res)
        ot_model = keras.models.load_model(overtrained_path)
        res = ot_model.evaluate(
            val_data_input,
            val_data_classes,
            verbose=1
        )
        print(op_model.metrics_names)
        print(res)



def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--verbose", type=int, help="Show training logs verbosely", default=0)
    parser.add_argument("--epoch-max", type=int, help="Maximum epochs to test to", default=1500)
    parser.add_argument("--error-rate", type=int, help="Error rate to use", default=10)
    parser.add_argument("--data-size", type=int, help="Size of train/test dataset", default=300)
    parser.add_argument("--prefix", help="Prefix for the experiment folder", default="")
    parser.add_argument("--num-nets", type=int, help="Number of component nets to do the experiment to", default=99)
    parser.set_defaults(func=main)
