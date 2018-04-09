"""
Given a dataset, and specs for input and output of a neural network, this module will attempt to
autotrain to optimization
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np


def autotrain(data: pd.DataFrame,
              test_data: pd.DataFrame,
              *,
              input_cols=("x", "y"),
              class_col="class",
              min_nodes=2,
              max_nodes=9,
              epoch_step=1,
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
    assert epoch_step > 0
    assert max_nodes >= min_nodes

    inputs = len(input_cols)
    outputs = data[class_col].nunique()

    train_data = data.as_matrix(columns=input_cols)
    test_data = data.as_matrix(columns=input_cols)

    loss_function = None
    if outputs == 2:
        outputs = 1
        loss_function = 'binary_crossentropy'
        train_classes = data["class"].values
        test_classes = data["class"].values
    else:
        loss_function = 'categorical_crossentropy'
        train_classes = keras.utils.to_categorical(data["class"].values, num_classes=outputs)
        test_classes = keras.utils.to_categorical(data["class"].values, num_classes=outputs)

    max_accuracy_so_far = 0
    max_accuracy_model = None

    for hidden_nodes in range(min_nodes, max_nodes+1):
        if verbose:
            print("Train For Hidden Nodes:", hidden_nodes)
        model = Sequential([
            Dense(hidden_nodes, input_shape=(inputs,), activation='sigmoid'),
            Dense(outputs, activation='sigmoid')
        ])

        model.compile(
            # TODO: Switch to binary_crossentropy if outputs are 2 and use a single output
            loss=loss_function,
            optimizer='sgd',
            metrics=['accuracy']
        )

        for learning_rate in np.arange(0.01, 0.5, 0.02):
            model.optimizer.lr = learning_rate
            if verbose:
                print(f"  With Learning Rate: {model.optimizer.lr:.4f}")

            trained_to_optimal_epoch = False
            next_epoch = epoch_step
            max_accuracy_with_this_learning_rate = 0
            epoch_of_max_accuracy_with_this_learning_rate = 0
            while not trained_to_optimal_epoch:
                hist = model.fit(
                    train_data,
                    train_classes,
                    verbose=0,
                    epochs=next_epoch,
                    initial_epoch=next_epoch-1
                )
                train_accuracy = hist.history["acc"][-1]
                loss, accuracy = model.evaluate(
                    test_data,
                    test_classes,
                    verbose=0,
                )
                # Test if there has been an accuracy improvement
                if accuracy > max_accuracy_with_this_learning_rate:
                    max_accuracy_with_this_learning_rate = accuracy
                    epoch_of_max_accuracy_with_this_learning_rate = next_epoch

                # Early exit conditions
                # if the last peak accuracy was more than 2000 steps ago,
                # or the last peak was within 45% of the start of the run,
                # and more than 300 epochs have elapsed
                if (((epoch_of_max_accuracy_with_this_learning_rate + 2000) < next_epoch
                     or
                     epoch_of_max_accuracy_with_this_learning_rate < (next_epoch * 0.45)
                    ) and next_epoch > 300):
                    if verbose:
                        print(f"    Found optimal epoch: {next_epoch}.\n"
                          f"      Params: nodes: {hidden_nodes}, lr: {learning_rate:.4f}, "
                          f"train accuracy: {train_accuracy:.4f} test accuracy: {accuracy:.4f}")
                    trained_to_optimal_epoch = True
                    if accuracy > max_accuracy_so_far:
                        max_accuracy_so_far = accuracy
                        max_accuracy_model = keras.models.clone_model(model)
                        max_accuracy_model.set_weights(model.get_weights())

                next_epoch += epoch_step

    # Todo - show some info about the final model
    if verbose:
        print(f"Testing Accuracy of trained Model: {max_accuracy_so_far:.4f}")

    # print("": max_accuracy_model)
    # TODO: Return a saved file rather than
    return max_accuracy_model
