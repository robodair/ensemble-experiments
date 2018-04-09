"""
Glue code for conducting experiment 1
"""
import argparse
import numpy as np

def main(args):
    print("Running Experiment")
    np.random.seed(args.seed)
    import ensemble_experiments.datagen2d as dg

    for error_rate in range(10, 50, 10):
        print(f"BEGIN RUN FOR ERROR RATE {error_rate}%")

        # generate data and split into test/train/validate
        data = dg.generate_data(300, error_rate, 2017)
        val_data = dg.generate_data(1000, error_rate, 2018)

        train_data = data[::2]
        test_data = data[1::2]

        for num_component_nets in range(1, 99):
            print(f" BEGIN RUN FOR {num_component_nets} NETS")
            train_test_data = []
            # Generate num_component_nets bags of data and train networks
            for i in range(num_component_nets):
                train_bag = train_data.sample(len(train_data), replace=True)
                test_bag = test_data.sample(len(test_data), replace=True)
                train_test_data.append((
                    train_bag, test_bag
                ))

            def train(x):
                import ensemble_experiments.autotrain as at
                return at.autotrain(*x, min_nodes=2, max_nodes=2, epoch_step=args.step, verbose=args.verbose)

            networks = list(map(train, train_test_data)) # TODO: Create a dask graph here, we can distribute the training

            # TODO: Run validation across component networks and aggregate results into predictions, compare with validation (real) classes
            print(networks)

        # network = at.autotrain(data, test_data, min_nodes=2, max_nodes=9, epoch_step=args.step, verbose=args.verbose)

def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--verbose", action='store_true', help="Show training logs verbosely")
    parser.add_argument("--step", type=int, help="Number of epochs between evaluations", default=1)
    parser.add_argument("--seed", type=int, help="Numpy randon seed", default=2018)
    parser.set_defaults(func=main)
