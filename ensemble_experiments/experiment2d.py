"""
Glue code for conducting experiment 1
"""
import os
# os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
# No verbose logging from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import time

import argparse
import signal
import numpy as np
from pathlib import Path
import pandas


def train(train_df, test_df, save_dir, epochs, verbose, net_number):
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
            patience=2000,
            verbose=verbose,
        )

        model = Sequential([
            Dense(20, input_shape=(2,), activation="sigmoid"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(lr=0.8),
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

    return {
        "id": net_number,
        "dir": save_dir,
        "net": save_net,
        "ot_net": save_overtrained_net
    }

def wrap_train(x):
    return train(*x)

def main(args):
    print("Running Experiment")

    import ensemble_experiments.datagen2d as dg

    print(f"BEGIN RUN FOR ERROR RATE {args.error_rate}%")
    ratedir = args.save_dir / f"error-{args.error_rate}"
    ratedir.mkdir(exist_ok=True, parents=True)

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

    nets = [] # list of dicts of paths to load

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
            train_bag = train_data.sample(len(train_data), replace=True)
            train_bag.to_csv(save_train)
        if save_test.exists():
            test_bag = pandas.read_csv(save_test)
        else:
            test_bag = test_data.sample(len(test_data), replace=True)
            test_bag.to_csv(save_test)

        print(f"GET NETWORK FOR {net_number}")
        nets.append(train(train_bag, test_bag, working_dir, args.epochs, args.verbose, net_number))

    # Run validation data through each net and save predictions for later analysis
    from keras.models import load_model
    val_data_xy = val_data.as_matrix(columns=('x', 'y'))

    for net_dict in nets:
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
            del net
            del predictions

            ot_net = load_model(net_dict["ot_net"])
            ot_predictions = pandas.DataFrame(
                data=ot_net.predict_classes(val_data_xy),
                columns=(f"ot-ANN-{net_dict['id']}",)
            )
            ot_predictions.to_csv(ot_validation_predictions_file)
            print(f"Saved OT Predictions for OT ANN {net_dict['id']} to {ot_validation_predictions_file}")
            del ot_net
            del ot_predictions
        else:
            print(f"Already have predictions for ANN {net_dict['id']}")


    # Work out which ANN's to use for each ANNE

    # TODO: Run validation across component networks and aggregate results into predictions
    # compare with validation (real) classes
    # print(f"Nets for ANNE: {network_paths}")

    # val_data_input = val_data.as_matrix(columns=('x', 'y'))
    # val_data_classes = val_data["realclass"].values
    # for net_path, overtrained_path in network_paths:
    #     op_model = keras.models.load_model(net_path)
    #     res = op_model.evaluate(
    #         val_data_input,
    #         val_data_classes,
    #         verbose=1
    #     )
    #     print(op_model.metrics_names)
    #     print(res)
    #     ot_model = keras.models.load_model(overtrained_path)
    #     res = ot_model.evaluate(
    #         val_data_input,
    #         val_data_classes,
    #         verbose=1
    #     )
    #     print(op_model.metrics_names)
    #     print(res)



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
    parser.set_defaults(func=main)
