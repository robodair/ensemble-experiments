"""
Glue code for conducting experiment 1
"""
import os
# os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
# No verbose logging from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import time
import random

import argparse
import signal
import numpy as np
from pathlib import Path
import pandas

from ensemble_experiments.datagen2d import CLASS_A, CLASS_B

def entropy_variance(stats: pandas.DataFrame, L: int):
    """
    Entropy Measure for Variance

    Parameters:
        stats: DataFrame with 'correct_count' and 'incorrect_count' columns detailing the number of
            networks that correctly and incorrectly classified each validation point respectively
        L: The number of networks in the ensemble
    """
    if L == 1:
        return 0
    N = len(stats)
    minimum_counts = stats.loc[:, ('correct_count', 'incorrect_count')].min(axis=1)
    coeff = 1/(L - np.ceil(L / 2))
    mult = coeff * minimum_counts
    summation = mult.sum()
    E = summation / N
    return E


def train(train_df, test_df, save_dir, epochs, verbose, net_number, learn_rate, hidden_nodes, patience):
    save_net = save_dir / "net.h5"
    save_overtrained_net = save_dir / "overtrained_net.h5"

    import keras
    from keras.callbacks import EarlyStopping
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD

    start_time = time.time()

    if save_net.exists() and save_overtrained_net.exists():
        print(f"Models already exist for {save_net}")
    else:
        print(f"Training for {save_net}")

        train_classes = train_df["class"].values
        test_classes = test_df["class"].values
        train_data = train_df.as_matrix(columns=("x", "y"))
        test_data = test_df.as_matrix(columns=("x", "y"))

        stopper = EarlyStopping(
            monitor="val_acc",
            patience=patience,
            verbose=verbose,
        )

        model = Sequential([
            Dense(hidden_nodes, input_shape=(2,), activation="sigmoid"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(lr=learn_rate),
            metrics=["accuracy"]
        )

        hist = model.fit(
            train_data,
            train_classes,
            verbose=verbose,
            epochs=epochs,
            validation_data=(test_data, test_classes),
            callbacks=[stopper]
        )
        model.save(save_net)

        op_epochs = len(hist.epoch)
        ot_epochs = op_epochs * 10

        print(f"Trained to {op_epochs} epochs as determined by early exit, saved to {save_net}\n"
              f"Beginning overtrain to {ot_epochs} epochs")

        train_time = time.time()
        print(f">> Train Time: {train_time-start_time:.2f} seconds")

        model.fit(
            train_data,
            train_classes,
            verbose=verbose,
            epochs=ot_epochs,
            initial_epoch=op_epochs,
            validation_data=(test_data, test_classes)
        )
        model.save(save_overtrained_net)

        print(f"Overtrained to {ot_epochs} epochs, saved to {save_overtrained_net}")

        end_time = time.time()
        print(f">> Overtrain Time: {end_time-train_time:.2f} seconds")
        print(f">>>> Total Time: {end_time-start_time:.2f} seconds")
    # Avoid Memory Leak https://github.com/keras-team/keras/issues/2102
    keras.backend.clear_session()

    return {
        "id": net_number,
        "dir": save_dir,
        "net": save_net,
        "ot_net": save_overtrained_net
    }


def main(args):
    print("Running Experiment")

    import ensemble_experiments.datagen2d as dg

    print(f"BEGIN RUN FOR ERROR RATE {args.error_rate}%")
    ratedir = args.save_dir / f"error-{args.error_rate}"
    ratedir.mkdir(exist_ok=True, parents=True)

    data_csv = ratedir / "data.csv"
    if not data_csv.exists():
        data = dg.generate_data(args.data_size, args.error_rate)
        data.to_csv(data_csv)
    else:
        data = pandas.read_csv(data_csv)

    val_data_csv = ratedir / "validation.csv"
    if not val_data_csv.exists():
        val_data = dg.generate_data(args.val_data_size, args.error_rate)
        val_data.to_csv(val_data_csv)
    else:
        val_data = pandas.read_csv(val_data_csv)

    train_data = data[::2]
    test_data = data[1::2]

    net_dicts = [] # list of dicts of paths to load

    for net_number in range(1, args.num_nets + 1):
        print(f"GET DATA FOR {net_number}")
        working_dir = ratedir / f"ANN-{net_number}"
        working_dir.mkdir(exist_ok=True)

        save_train = working_dir / "train.csv"
        save_test = working_dir / "test.csv"
        working_dir.mkdir(exist_ok=True)
        if save_train.exists():
            train_bag = pandas.read_csv(save_train)
        else:
            if not args.control:
                train_bag = train_data.sample(len(train_data), replace=True)
            else:
                train_bag = train_data
            train_bag.to_csv(save_train)
        if save_test.exists():
            test_bag = pandas.read_csv(save_test)
        else:
            if not args.control:
                test_bag = test_data.sample(len(test_data), replace=True)
            else:
                test_bag = test_data
            test_bag.to_csv(save_test)

        print(f"GET NETWORK FOR {net_number}")
        net_dicts.append(
            train(train_bag, test_bag, working_dir, args.epochs,
                  args.verbose, net_number, args.learn_rate, args.hidden_nodes, args.patience)
        )

    # Run validation data through each net and save predictions for later analysis
    from keras.models import load_model
    import keras
    val_data_xy = val_data.as_matrix(columns=('x', 'y'))

    for net_dict in net_dicts:
        validation_predictions_file = net_dict["dir"] / "val_predictions.csv"
        ot_validation_predictions_file = net_dict["dir"] / "overtrained_val_predictions.csv"

        if not validation_predictions_file.exists() or not ot_validation_predictions_file.exists():
            net = load_model(net_dict["net"])
            predictions = pandas.DataFrame(
                data=net.predict_classes(val_data_xy),
                columns=(f"ANN-{net_dict['id']}",)
            )
            predictions.to_csv(validation_predictions_file)
            print(f"Saved Predictions for ANN {net_dict['id']} to {validation_predictions_file}")

            ot_net = load_model(net_dict["ot_net"])
            ot_predictions = pandas.DataFrame(
                data=ot_net.predict_classes(val_data_xy),
                columns=(f"ot-ANN-{net_dict['id']}",)
            )
            ot_predictions.to_csv(ot_validation_predictions_file)
            print(f"Saved OT Predictions for OT ANN {net_dict['id']} to {ot_validation_predictions_file}")
        else:
            print(f"Already have predictions for ANN {net_dict['id']}")

        # Avoid Memory Leak https://github.com/keras-team/keras/issues/2102
        keras.backend.clear_session()

    all_stats = []
    # Work out which ANN's to use for each ANNE
    annes_stats_file = ratedir / "anne_stats.csv"
    if not annes_stats_file.exists():
        for anne_number in range(1, args.num_ensembles+1):
        # for anne_number in range(45, args.num_ensembles+1):
            print(f"Producing stats for EarlyExit and Overtrained ANNE {anne_number}")
            # Select the networks to use (e.g. for ANNE 4 choose 4 random numbers between 1 and args.num_nets)
            net_predictions = pandas.DataFrame()
            ot_net_predictions = pandas.DataFrame()
            # for simplicity, truth dataframe column names match prediction column names
            truth = pandas.DataFrame()
            ot_truth = pandas.DataFrame()

            chosen_components_file = ratedir / f"anne-{anne_number}-component-nets.csv"
            if chosen_components_file.exists():
                anne_components = pandas.read_csv(chosen_components_file)['nets']
            else:
                df = pandas.DataFrame({
                    'nets': random.sample(range(1, args.num_nets + 1), anne_number)
                })
                df.to_csv(chosen_components_file)
                anne_components = df['nets']

            for net_number in anne_components:

                net_name = f"ANN-{net_number}"
                ot_net_name = f"ot-ANN-{net_number}"
                directory = ratedir / net_name
                validation_predictions_file = directory / "val_predictions.csv"
                ot_validation_predictions_file = directory / "overtrained_val_predictions.csv"

                net_predictions[net_name] = pandas.read_csv(
                    validation_predictions_file)[net_name]
                truth[net_name] = val_data["realclass"]
                ot_net_predictions[ot_net_name] = pandas.read_csv(
                    ot_validation_predictions_file)[ot_net_name]
                ot_truth[ot_net_name] = val_data["realclass"]

            # Prediction Info
            anne_prediction_stats = pandas.DataFrame({
                "anne_prediction": net_predictions.mode(axis=1)[0].astype(int),
                "correct_count": (net_predictions == truth).sum(axis=1),
                "incorrect_count": (net_predictions != truth).sum(axis=1),
            })

            anne_ot_prediction_stats = pandas.DataFrame({
                "anne_prediction": ot_net_predictions.mode(axis=1)[0].astype(int),
                "correct_count": (ot_net_predictions == ot_truth).sum(axis=1),
                "incorrect_count": (ot_net_predictions != ot_truth).sum(axis=1),
            })

            # Variance computation
            variance = entropy_variance(anne_prediction_stats, anne_number)
            ot_variance = entropy_variance(anne_ot_prediction_stats, anne_number)
            print("   Early Exit Entropy Variance", variance)
            print("  Overtrained Entropy Variance", ot_variance)

            # Accuracy Computation
            accuracy = (anne_prediction_stats['anne_prediction'] == val_data["realclass"]).sum() / len(val_data) * 100
            ot_accuracy = (anne_ot_prediction_stats['anne_prediction'] == val_data["realclass"]).sum() / len(val_data) * 100
            print("   Early Exit Accuracy", accuracy)
            print("  Overtrained Accuracy", ot_accuracy)

            all_stats.append({
                'anne_number': anne_number,
                'accuracy': accuracy,
                'ot_accuracy': ot_accuracy,
                'entropy_var': variance,
                'ot_entropy_var': ot_variance
            })

        net_stats = pandas.DataFrame(all_stats)
        net_stats.to_csv(annes_stats_file)
    else:
        net_stats = pandas.read_csv(annes_stats_file)


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("save_dir",
                        help="Directory in which to save experiment files", type=Path)
    parser.add_argument("-v", "--verbose",
                        type=int, help="Show training logs verbosely", default=0)
    parser.add_argument("-e", "--epochs",
                        type=int, help="Maximum epochs to early exit train to", default=20000)
    parser.add_argument("-r", "--error-rate",
                        type=int, help="Data error rate to use", default=10)
    parser.add_argument("-d", "--data-size",
                        type=int, help="Size of train/test dataset", default=300)
    parser.add_argument("-c", "--num-nets",
                        type=int, help="Size of component network pool",
                        default=300)
    parser.add_argument("-n", "--num-ensembles",
                        type=int, help="Number of ensembles to test",
                        default=99)
    parser.add_argument("-z", "--val-data-size",
                        type=int, help="Number of points to use for validation",
                        default=3000)
    parser.add_argument("-l", "--learn-rate",
                        type=float, help="Network learn rate",
                        default=0.8)
    parser.add_argument("--hidden-nodes",
                        type=int, help="Number of hidden nodes to use for the networks",
                        default=20)
    parser.add_argument("-p", "--patience",
                        type=int, help="Patience (epochs) for early exit", default=2000)
    parser.add_argument("--control", action="store_true",
                        help="Conduct a control experiment. No bagging will be done."
                             "The initial weights will be the only thing that make the networks diverse")
    parser.set_defaults(func=main)
