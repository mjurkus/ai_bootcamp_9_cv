import copy
import math
from typing import Tuple, Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from transformations import ImageTransformation


class ImageDatasetConfig:

    def __init__(
            self,
            img_dims: Tuple[int, int, int],
            preprocess_pipeline: List[ImageTransformation] = [],
            batch_size: int = 8,
            shuffle: bool = False,
    ):
        self.img_dims = img_dims
        self.preprocess_pipeline = preprocess_pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle

    def copy(
            self,
            preprocess_pipeline: List[ImageTransformation],
            overwrite_pipeline: bool = False,
            shuffle: bool = False,
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
        self.label_map = {}

    def build_from_df(self, df: pd.DataFrame, path_col: str, label_col: Optional[str] = None) -> 'ImageDataset':
        labels = df[label_col].values if label_col else np.empty((1, 1))
        return self._build(df[path_col].values, labels)

    def with_label_map(self, label_map: Dict[str, int]) -> 'ImageDataset':
        self.label_map = label_map
        return self

    def _build(self, x: np.ndarray, y: np.ndarray) -> 'ImageDataset':
        self.x = x
        self.y = y
        self.length = len(x)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.steps = math.ceil(self.length / self.config.batch_size)
        self.label_map = self.label_map or {value: key for key, value in dict(enumerate(np.unique(self.y))).items()}

        image_ds = tf.data.Dataset.from_tensor_slices(x)

        for fun in self.config.preprocess_pipeline:
            image_ds = image_ds.map(fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        label_ds = tf.data.Dataset.from_tensor_slices(np.asarray([self.label_map[label] for label in self.y]))
        dataset = tf.data.Dataset.zip((image_ds, label_ds))

        if self.config.shuffle:
            dataset = dataset.shuffle(self.config.batch_size)

        self.data = dataset.batch(self.config.batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return self

    def show(self, cols: int = 8, batches: int = 1) -> None:
        labels = {key: value for key, value in enumerate(self.label_map)}

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
                ax[idx].set_title(f"{y} :: {labels[y]}")
                i += 1
