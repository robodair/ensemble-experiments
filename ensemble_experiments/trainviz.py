"""
Visualize the training of an ANN, to help in picking a reasonable number of hidden nodes and
learning rate. An early exit callback stops training after 2000 epochs
"""
import argparse
import time
from pathlib import Path
import sys

import numpy as np

import matplotlib.pyplot as pyplot

from keras.callbacks import Callback

from ensemble_experiments.datagen2d import generate_data, CLASS_A, CLASS_B, generate_uniform_data
import ensemble_experiments.dataviz as dv

def plot_visualisation(vis_data, train_data, title):
    """
    Plot visualisation data from, then call dataviz to overlay with train dataset
    """
    pyplot.clf()
    ax = pyplot.subplot(111)

    # Class A points
    pyplot.scatter(
        vis_data.loc[vis_data["class"] == CLASS_A, "x"],
        vis_data.loc[vis_data["class"] == CLASS_A, "y"],
        c="pink",
        marker=".",
        label="Classified as A"
    )

    # Class B points
    pyplot.scatter(
        vis_data.loc[vis_data["class"] == CLASS_B, "x"],
        vis_data.loc[vis_data["class"] == CLASS_B, "y"],
        c="skyblue",
        marker=".",
        label="Classified as B"
    )

    if train_data is not None:
        dv.plot_dataset(train_data)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.title(title)

    pyplot.pause(0.1)


class PlotDiscriminatorCallback(Callback):
    """
    Keras Callback for plotting at specified epoch increments
    """

    def __init__(self, delta, model, vis_df, train_df, start_time, title):
        self.delta = delta
        self.model = model
        self.vis_df = vis_df
        self.train_df = train_df
        self.start_time = start_time
        self.title = title

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.delta == 0:
            print(f"Plotting for epoch {epoch}, val_acc {logs['val_acc'] * 100:.3f}, at "
                  f"{time.time() - self.start_time:.2f} secs")
            self.vis_df["class"] = self.model.predict_classes(
                self.vis_df.as_matrix(columns=("x", "y")))
            plot_visualisation(self.vis_df, self.train_df, self.title)



def main(args):
    from keras.models import Sequential, load_model
    from keras.layers import Dense
    from keras.optimizers import SGD
    from keras.callbacks import EarlyStopping

    pyplot.ion()

    start_time = time.time()

    # Gen dataset of specified size
    data = generate_data(args.data_size, args.error_rate, args.seed)
    train_df = data[::2]
    test_df = data[1::2]
    train_classes = train_df["class"].values
    test_classes = test_df["class"].values
    train_data = train_df.as_matrix(columns=("x", "y"))
    test_data = test_df.as_matrix(columns=("x", "y"))

    # Gen uniform dataset for plotting distriminator
    vis_df = generate_uniform_data()

    if args.show_net is not None:
        model = load_model(args.show_net / 'net.h5')
        ot_model = load_model(args.show_net / 'overtrained_net.h5')
        vis_df["class"] = model.predict_classes(vis_df.as_matrix(columns=("x", "y")),
                                            verbose=args.verbose)
        plot_visualisation(vis_df, None, f"Optimal")
        pyplot.figure()
        vis_df["class"] = ot_model.predict_classes(vis_df.as_matrix(columns=("x", "y")),
                                            verbose=args.verbose)
        plot_visualisation(vis_df, None, f"Overtrained")
        pyplot.show(block=True) # block till window is closed
        sys.exit()



    stopper = EarlyStopping(
        monitor="val_acc",
        patience=args.patience,
        verbose=args.verbose,
    )

    model = Sequential([
        Dense(args.hidden_nodes, input_shape=(2,), activation="sigmoid"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=SGD(lr=args.learn_rate),
        metrics=["accuracy"]
    )

    callbacks = [
        PlotDiscriminatorCallback(
            args.plot_delta, model, vis_df, train_df, start_time,
            f'Optimal - {args.error_rate}% Data Error'
        ),
        stopper
    ]
    if args.quiet:
        callbacks = [stopper]

    hist = model.fit(
        train_data,
        train_classes,
        verbose=args.verbose,
        epochs=args.epochs,
        validation_data=(test_data, test_classes),
        callbacks=callbacks
    )

    op_epochs = len(hist.epoch)

    test_classes_real = test_df["class"].values

    op_metrics = dict(zip(model.metrics_names, model.evaluate(test_data, test_classes_real,
                                                              verbose=args.verbose)))

    pyplot.title(f"Optimal - {args.error_rate}% Data Error, "
                 f"{op_epochs} Epochs, {op_metrics['acc'] * 100:.2f}% Accuracy")
    if args.save_dir:
        fig_name = f"{args.error_rate}-error_optimal_seed-{args.seed}_{time.strftime('%Y-%m-%d-%H-%M-%S')}.png"
        pyplot.savefig(args.save_dir / fig_name)

    pyplot.figure() # New figure for Overtrain plotting
    pyplot.pause(1)

    ot_epochs = op_epochs * 10
    print(f"Beginning Overtraining to {ot_epochs} epochs")

    callbacks = [
        PlotDiscriminatorCallback(
            args.plot_delta * 5, model, vis_df, train_df, start_time,
                f'Overtrained - {args.error_rate}% Data Error'
            )
    ]
    if args.quiet:
        callbacks = []

    hist2 = model.fit(
        train_data,
        train_classes,
        verbose=args.verbose,
        epochs=ot_epochs,
        initial_epoch=op_epochs,
        validation_data=(test_data, test_classes),
        callbacks=callbacks
    )

    vis_df["class"] = model.predict_classes(vis_df.as_matrix(columns=("x", "y")),
                                            verbose=args.verbose)

    print("Close Plots to exit")

    ot_metrics = dict(zip(model.metrics_names, model.evaluate(test_data, test_classes_real,
                                                              verbose=args.verbose)))

    plot_visualisation(vis_df, train_df,
        f"Overtrained - {args.error_rate}% Data Error, {ot_epochs} Epochs, {ot_metrics['acc']*100:.2f}% Accuracy")

    if args.save_dir:
        fig_name = f"{args.error_rate}-error_overtrained_seed-{args.seed}_{time.strftime('%Y-%m-%d-%H-%M-%S')}.png"
        pyplot.savefig(args.save_dir / fig_name)

    pyplot.pause(0.1)
    pyplot.show(block=True) # block till window is closed


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--show-net",
                        type=Path, help="Show the discriminators of the nets in the given directory")
    parser.add_argument("-v","--verbose",
                        type=int, help="Show training logs verbosely", default=0)
    parser.add_argument("-e","--epochs",
                        type=int, help="Maximum epochs to test to", default=100000)
    parser.add_argument("-x","--plot-delta",
                        type=int, help="Plot after this many epochs", default=500)
    parser.add_argument("-r","--error-rate",
                        type=int, help="Error rate to use", default=10)
    parser.add_argument("-d", "--data-size",
                        type=int, help="Size of train/test dataset", default=300)
    parser.add_argument("-l", "--learn-rate",
                        type=float, help="SGD Learning Rate to use", default=0.8)
    parser.add_argument("-n", "--hidden-nodes",
                        type=int, help="Num Hidden Nodes", default=20)
    parser.add_argument("-s", "--seed",
                        type=int, help="Seed for dataset generation", default=9)
    parser.add_argument("-p", "--patience",
                        type=int, help="Patience (epochs) for early exit", default=2000)
    parser.add_argument("--save-dir",
                        type=Path, help="Directory in which to save optimal and overtrained plots")
    parser.add_argument("-q", "--quiet",
                        action="store_true", help="Don't show the plots")

    parser.set_defaults(func=main)
