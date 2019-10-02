from pathlib import Path
import numpy as np
import pandas as pd
import os
import re
import math
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, List
from transformations import ImageTransformation


class ImageDatasetConfig:

    def __init__(
            self,
            img_dims: Tuple[int, int, int],
            parallel_calls: int = 1,
            prefetch: int = 1,
            preprocess_pipeline: List[ImageTransformation] = [],
            batch_size: int = 8,
            shuffle: bool = False,
    ):
        self.parallel_calls = parallel_calls
        self.img_dims = img_dims
        self.prefetch = prefetch
        self.preprocess_pipeline = preprocess_pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle

    def copy(self,
             preprocess_pipeline: List[ImageTransformation],
             overwrite_pipeline: bool = False,
             shuffle: bool = False
             ) -> 'ImageDatasetConfig':
        new = copy.deepcopy(self)
        new.shuffle = shuffle

        if overwrite_pipeline:
            new.preprocess_pipeline = preprocess_pipeline
        else:
            new.preprocess_pipeline += preprocess_pipeline

        return new


class ImageDataset:
    data: tf.data.Dataset
    x: np.ndarray
    y: np.ndarray
    length: int
    steps: int
    classes: np.ndarray
    n_classes: int

    def __init__(self, config: ImageDatasetConfig):
        self.config = config

    def build_from_df(self, df: pd.DataFrame, path_col: str, label_col: Optional[str] = None) -> 'ImageDataset':
        labels = df[label_col].values if label_col else np.empty((1, 1))
        return self.__build(df[path_col].values, labels)

    def build_from_path(self, path: Path, regexp: str, default_label: Optional[str] = None) -> 'ImageDataset':
        paths = []
        labels = []

        for value in os.listdir(str(path)):
            match = re.match(regexp, value)
            if match:
                labels.append(match.group(1))
            elif default_label:
                labels.append(default_label)
            else:
                raise ValueError(f"No match found and no default value provided for value: {value}")

            paths.append(f"{path}/{value}")

        return self.__build(np.asarray(paths), np.asarray(labels))

    def __build(self, x: np.ndarray, y: np.ndarray) -> 'ImageDataset':
        self.x = x
        self.y = y
        self.length = len(x)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.steps = math.ceil(self.length / self.config.batch_size)

        image_ds = tf.data.Dataset.from_tensor_slices(x)

        for fun in self.config.preprocess_pipeline:
            image_ds = image_ds.map(fun, num_parallel_calls=self.config.parallel_calls)

        label_ds = tf.data.Dataset.from_tensor_slices(y.astype(float))
        dataset = tf.data.Dataset.zip((image_ds, label_ds))

        if self.config.shuffle:
            dataset = dataset.shuffle(self.config.batch_size)

        self.data = dataset.batch(self.config.batch_size).repeat().prefetch(self.config.prefetch)

        return self

    def show(self, cols: int = 8, batches: int = 1) -> None:
        if cols >= self.config.batch_size * batches:
            cols = self.config.batch_size * batches
            rows = 1
        else:
            rows = math.ceil(self.config.batch_size * batches / cols)
        _, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        i = 0
        for x_batch, y_batch in self.data.take(batches):
            for (x, y) in zip(x_batch.numpy(), y_batch.numpy()):
                idx = (i // cols, i % cols) if rows > 1 else i % cols
                ax[idx].axis("off")
                ax[idx].imshow(x)
                ax[idx].set_title(y)
                i += 1
