"""
Visualize the training of an ANN, to help in picking a reasonable number of hidden nodes and epoch
"""
import argparse
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import Callback

CLASS_A = 0
CLASS_B = 1

import ensemble_experiments.datagen2d as dg
import ensemble_experiments.plotutils as plotutils

def main(args):
    start_time = time.time()

    # Gen dataset of specified size
    data = dg.generate_data(args.data_size, args.error_rate, args.seed)
    train_df = data[::2]
    test_df = data[1::2]
    train_classes = train_df["class"].values
    test_classes = test_df["class"].values
    train_data = train_df.as_matrix(columns=('x', 'y'))
    test_data = test_df.as_matrix(columns=('x', 'y'))
    # Gen uniform dataset for plotting distriminator
    vis_df = dg.generate_visualization_data()

    # Train network, plotting at each plot_epochs until complete
    class PlotDiscriminatorCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % args.plot_delta == 0:
                print(f"Plotting for epoch {epoch}, acc {logs['acc'] * 100:.3f}")
                vis_df['class'] = model.predict_classes(vis_df.as_matrix(columns=('x', 'y')))
                plotutils.plot_discriminator_visualisation(vis_df, train_df)

    model = Sequential([
        Dense(args.hidden_nodes, input_shape=(2,), activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=SGD(lr=args.learn_rate),
        metrics=['accuracy']
    )
    hist = model.fit(
        train_data,
        train_classes,
        verbose=args.verbose,
        epochs=args.epochs,
        validation_data=(test_data, test_classes),
        batch_size=len(train_data),
        callbacks=[PlotDiscriminatorCallback()]
    )

def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--verbose", type=int, help="Show training logs verbosely", default=0)
    parser.add_argument("--epochs", type=int, help="Maximum epochs to test to", default=50000)
    parser.add_argument("--plot-delta", type=int, help="Plot after this many epochs", default=5000)
    parser.add_argument("--error-rate", type=int, help="Error rate to use", default=10)
    parser.add_argument("--data-size", type=int, help="Size of train/test dataset", default=300)
    parser.add_argument("-lr", "--learn-rate", type=float, help="Learning Rate to Use", default=0.2)
    parser.add_argument("--hidden-nodes", type=int, help="Num Hidden Nodes", default=12)
    parser.add_argument("--seed", type=int, help="Seed for dataset generation", default=9)

    parser.set_defaults(func=main)
