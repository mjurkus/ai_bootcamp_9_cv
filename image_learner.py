import shutil
from abc import ABC
from pathlib import Path
from typing import Tuple, Any, List

import numpy as np
import pandas as pd
from tensorflow import keras

from data import DataContainer


class BaseLearner(ABC):
    model: keras.Model
    history: keras.callbacks.History

    def __init__(self, model_path: Path, data: DataContainer) -> None:
        self.data = data
        self.model_path = model_path
        self.weights_path = self.model_path / 'weights.h5'
        self.architecture_path = self.model_path / 'model.json'
        self.logs_path = self.model_path / 'logs'

    def save(self) -> None:
        self.model.save_weights(str(self.weights_path))
        with open(str(self.architecture_path), "w") as f:
            f.write(self.model.to_json())

    def load(self, weights_only: bool = False) -> None:
        if weights_only:
            print(f'Loading weights only from {self.weights_path}')
            self.model.load_weights(str(self.weights_path))
        else:
            print(f'Loading architecture and weights from {self.model_path}')
            with open(str(self.architecture_path), "r") as f:
                self.model = keras.models.model_from_json(f.read())

            self.model.load_weights(str(self.weights_path))

        print('Model loaded successfully')

    def compile(self, optimizer: Any, lr: float, loss: Any, metrics: Any) -> None:
        raise NotImplementedError()

    def auto_train(self, epochs: int, easing_epochs: int, optimizer: Any, lr: float, loss: Any,
                   metrics: List[Any]) -> None:
        raise NotImplementedError()

    def train(self, epochs: int) -> None:
        reduce_lr_patience = max(2, epochs // 3)
        early_stopping_patience = reduce_lr_patience * 2

        self.history = self.model.fit(
            x=self.data.train.data,
            steps_per_epoch=self.data.train.steps,
            validation_data=self.data.validation.data,
            validation_steps=self.data.validation.steps,
            epochs=epochs,
            callbacks=[
                keras.callbacks.ModelCheckpoint(str(self.weights_path), save_best_only=True, save_weights_only=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=reduce_lr_patience),
                keras.callbacks.EarlyStopping(patience=early_stopping_patience, restore_best_weights=True),
            ],
            verbose=1,
        )

    def predict_dataset(self, mode: str = 'validation', verbose: int = 0) -> np.ndarray:
        dataset = getattr(self.data, mode)
        return self.model.predict(dataset.data, steps=dataset.steps, verbose=verbose)

    def evaluate_dataset(self, mode: str = 'validation', verbose: int = 0) -> np.ndarray:
        dataset = getattr(self.data, mode)
        return self.model.evaluate(dataset.data, steps=dataset.steps, verbose=verbose)


class ImageLearner(BaseLearner):

    def __init__(self,
                 model_path: Path,
                 data: DataContainer,
                 base_model: Any,
                 input_shape: Tuple[int, int, int],
                 dropout: float = 0.0,
                 l1: float = 1e-8,
                 l2: float = 1e-8,
                 override: bool = False,
                 load: bool = True
                 ) -> None:
        super().__init__(model_path, data)
        self.n_classes = data.train.n_classes
        self.input_shape = input_shape
        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2

        self.base_model = base_model(include_top=False, input_shape=input_shape)
        x = keras.layers.concatenate(
            [
                keras.layers.GlobalAvgPool2D()(self.base_model.output),
                keras.layers.GlobalMaxPool2D()(self.base_model.output),
            ]
        )
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Dense(
            self.n_classes,
            kernel_regularizer=keras.regularizers.l1_l2(l1, l2),
            activation=keras.activations.sigmoid,
        )(x)

        self.model = keras.Model(inputs=self.base_model.inputs, outputs=x)

        if self.model_path.exists():
            print('Existing model data path exists')
            if load:
                self.load()
            elif override:
                try:
                    shutil.rmtree(str(model_path))
                except OSError as err:
                    print(f"Error while deleting {model_path} directory. {err}")

                self.model_path.mkdir(parents=True, exist_ok=True)
        else:
            self.model_path.mkdir(parents=True, exist_ok=True)

        self.save()

    def compile(self, optimizer: Any, lr: float, loss: Any, metrics: List[Any]) -> None:
        self.model.compile(
            optimizer=optimizer(lr),
            loss=loss,
            metrics=metrics
        )

    def freeze(self) -> None:
        print("Freezing all except last model layers")
        for layer in self.model.layers[:-1]:
            layer.trainable = False

    def unfreeze(self) -> None:
        print("Unfreezing all layers")
        for layer in self.model.layers:
            layer.trainable = True

    def auto_train(self, epochs: int, easing_epochs: int, optimizer: Any, lr: float, loss: Any,
                   metrics: List[Any]) -> None:
        if easing_epochs:
            self.freeze()
            self.compile(optimizer=optimizer, lr=lr, loss=loss, metrics=metrics)

            print("Training frozen model")
            self.train(easing_epochs)
            self.load(weights_only=True)
            self.unfreeze()
            print("Finished training frozen model")

        self.compile(optimizer=optimizer, lr=lr, loss=loss, metrics=metrics)

        print("Starting model training")
        self.train(epochs)
        self.load(weights_only=True)
        print("Model training completed")

    def show_history(self, contains: str, skip: int = 0) -> None:
        history_df = pd.DataFrame(self.history.history)
        history_df[list(history_df.filter(regex=contains))].iloc[skip:].plot()
