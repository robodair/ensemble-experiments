"""
Given a dataset, and specs for input and output of a neural network, this module will attempt to
autotrain to optimization
"""
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np

def epoch_optimise(original_model, max_epochs, train_data, train_classes, test_data, test_classes, compile_args, *, verbose=True, lr=0.1, return_optimum=False):

    start_time = time.time()
    model = keras.models.clone_model(original_model) # Avoid mutating original
    model.compile(**compile_args)
    model.optimizer.lr = lr
    compile_time = time.time()

    batch_size = len(train_data) # Data all fits in memory (I hope)

    hist = model.fit(
        train_data,
        train_classes,
        verbose=0,
        epochs=max_epochs,
        initial_epoch=0,
        validation_data=(test_data, test_classes),
        batch_size=batch_size
    )
    fit_time = time.time()
    test_accuracies = hist.history["val_acc"]
    index = np.argmax(test_accuracies)
    accuracy = test_accuracies[index]
    accuracy_time = time.time()
    if verbose:
        print(f"      Accuracy {accuracy * 100:.4f}% at Epoch {index}\n"
              f"      C: {compile_time - start_time}s, F: {fit_time - compile_time}, A: {accuracy_time - fit_time}s")

    if return_optimum:
        # Train model to optimum epochs
        max_accuracy_model = keras.models.clone_model(original_model) # Avoid mutating original
        model.compile(**compile_args)
        model.optimizer.lr = lr
        hist = model.fit(
            train_data,
            train_classes,
            verbose=0,
            epochs=index,
            initial_epoch=0,
            batch_size=batch_size
        )
    else:
        max_accuracy_model = None

    return max_accuracy_model, index, accuracy


def autotrain(data: pd.DataFrame,
              test_data: pd.DataFrame,
              *,
              input_cols=("x", "y"),
              class_col="class",
              min_nodes=1,
              max_nodes=9,
              epoch_max=3000,
              verbose=False
    ) -> Sequential:
    """
    Accept parameters and construct a simple MLP network (one hidden layer, sigmoid activation),
    autotrain to approximate optimisation.

    Autotraining is done with variations to the number of hidden nodes, the learning rate, and
    finding the best epoch with each combination.

    Arguments:
        data: dataframe with input cols and class column to use for training
        test_data: dataframe with input cols and class column to use for testing
        classes: class labels

    Returns:
        The trained model, and a copy of the model overtrained by 2000 epochs
        if also_overtrain is True
    """
    start_time = time.time()
    assert epoch_max > 0
    assert max_nodes >= min_nodes

    inputs = len(input_cols)
    outputs = data[class_col].nunique()

    train_data = data.as_matrix(columns=input_cols)
    test_data = data.as_matrix(columns=input_cols)

    if outputs == 2:
        outputs = 1
        loss_function = 'binary_crossentropy'
        train_classes = data["class"].values
        test_classes = data["class"].values
        optimizer = 'rmsprop'
    else:
        loss_function = 'categorical_crossentropy'
        train_classes = keras.utils.to_categorical(data["class"].values, num_classes=outputs)
        test_classes = keras.utils.to_categorical(data["class"].values, num_classes=outputs)
        optimizer = 'sgd'

    compile_args = dict(
        loss=loss_function,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    # Determine the best number of hidden nodes to use
    hidden_nodes_max_accuracy = 0
    best_num_hidden_nodes = None
    for hidden_nodes in range(min_nodes, max_nodes+1):
        if verbose:
            print("    Train For Hidden Nodes:", hidden_nodes)
        model = Sequential([
            Dense(hidden_nodes, input_shape=(inputs,), activation='sigmoid'),
            Dense(outputs, activation='sigmoid')
        ])
        _, _, accuracy = epoch_optimise(model, epoch_max, train_data, train_classes, test_data, test_classes, compile_args, verbose=verbose)
        if accuracy > hidden_nodes_max_accuracy:
            best_num_hidden_nodes = hidden_nodes
    if verbose:
        print(f"Found best hidden nodes {best_num_hidden_nodes}")

    model = Sequential([
        Dense(best_num_hidden_nodes, input_shape=(inputs,), activation='sigmoid'),
        Dense(outputs, activation='sigmoid')
    ])

    best_lr = None
    best_epochs = None
    best_accuracy = 0
    best_model = None
    for lr in np.concatenate([np.arange(0.1, 0.4, 0.05), np.arange(0.09, 0.04, -0.01)]):
        if verbose:
            print(f"    Testing LR: {lr}")
        returned_model, epochs, accuracy = epoch_optimise(model, epoch_max, train_data, train_classes, test_data, test_classes, compile_args, verbose=verbose, lr=lr, return_optimum=True)
        if accuracy > best_accuracy:
            best_epochs = epochs
            best_accuracy = accuracy
            best_lr = lr
            best_model = returned_model

    end_time = time.time()
    if verbose:
        print(f"    Epoch: {best_epochs}.\n Nodes: {best_num_hidden_nodes},"
              f"    LR: {best_lr:.4f}, Acc: {best_accuracy:.4f}\n"
              f"    Time {end_time - start_time}s")

    return best_model
