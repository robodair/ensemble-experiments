"""
Given a dataset, and specs for input and output of a neural network, this module will attempt to
autotrain to optimization
"""
import time
from pathlib import Path
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np

def autotrain(data: pd.DataFrame,
              test_data: pd.DataFrame,
              model_folder: Path,
              *,
              min_nodes: int = 2,
              max_nodes: int = 9,
              max_epochs: int = 1000,
              input_cols: tuple = ("x", "y"),
              class_col: str = "class",
              verbose: int = 1
    ) -> Sequential:

    def log(string):
        # if verbose:
        print("> autotrain2.autotrain >> ", string)

    assert max_nodes >= min_nodes

    inputs = len(input_cols)
    outputs = data[class_col].nunique()

    train_data = data.as_matrix(columns=input_cols)
    test_data = test_data.as_matrix(columns=input_cols)

    if outputs == 2:
        outputs = 1
        loss_function = 'binary_crossentropy'
        train_classes = data["class"].values
        test_classes = data["class"].values
    else:
        loss_function = 'categorical_crossentropy'
        train_classes = keras.utils.to_categorical(data["class"].values, num_classes=outputs)
        test_classes = keras.utils.to_categorical(data["class"].values, num_classes=outputs)

    stopper = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=verbose,
        min_delta=0.001
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=30,
        min_lr=0.006,
        cooldown=50
    )

    models = []
    for hidden_nodes in range(min_nodes, max_nodes+1):
        model = Sequential([
            Dense(hidden_nodes, input_shape=(inputs,), activation='sigmoid'),
            Dense(outputs, activation='sigmoid')
        ])
        model.compile(
            loss=loss_function,
            optimizer='sgd',
            metrics=['accuracy']
        )
        hist = model.fit(
            train_data,
            train_classes,
            verbose=verbose,
            epochs=max_epochs,
            validation_data=(test_data, test_classes),
            batch_size=len(train_data),
            callbacks=[reduce_lr, stopper]
        )
        models.append({
            "model": model,
            "acc": hist.history['val_acc'][-1],
            "epochs": len(hist.history['val_acc'])
        })

    op_model_dict = max(models, key=lambda x: x["acc"])
    op_model = op_model_dict["model"]
    op_model.save(model_folder / "op_model.hdf5")

    ot_model = keras.models.clone_model(op_model)
    ot_model.compile(
        loss=loss_function,
        optimizer='sgd',
        metrics=['accuracy']
    )
    hist = ot_model.fit(
            train_data,
            train_classes,
            verbose=verbose,
            epochs=op_model_dict["epochs"] * 10,
            validation_data=(test_data, test_classes),
            batch_size=len(train_data),
            callbacks=[reduce_lr, stopper]
    )
    ot_model.save(model_folder / "ot_model.hdf5")
    return op_model, ot_model