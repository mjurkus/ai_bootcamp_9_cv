from abc import ABC, abstractmethod
from typing import Any, Tuple

import tensorflow as tf


class ImageTransformation(ABC):

    def __call__(self, arg: Any) -> Any:
        return self.transform(arg)

    @abstractmethod
    def transform(self, arg: Any) -> Any:
        raise NotImplementedError("Image transformation is not implemented")


class ImageParser(ImageTransformation):

    def transform(self, filename: str) -> tf.Tensor:
        image = tf.image.decode_jpeg(
            tf.io.read_file(filename), channels=3
        )

        return tf.image.convert_image_dtype(image, tf.float32)


class ImageLRFlpTransformation(ImageTransformation):

    def transform(self, image: tf.Tensor) -> tf.Tensor:
        return tf.image.flip_left_right(image)


class ImageUDFlpTransformation(ImageTransformation):

    def transform(self, image: tf.Tensor) -> tf.Tensor:
        return tf.image.flip_up_down(image)


class CropTransformation(ImageTransformation, ABC):

    def __init__(self, image_dimensions: Tuple[int, int, int], crop_adjustment: float = 1.0):
        self.image_dimensions = image_dimensions
        self.crop_adjustment = crop_adjustment
        self.height, self.width, _ = image_dimensions


class ImageCropTransformation(CropTransformation):

    def transform(self, image: tf.Tensor) -> tf.Tensor:
        crop_height, crop_width = [
            int(x * self.crop_adjustment) for x in (self.height, self.width)
        ]
        image = tf.image.resize(image, (crop_height, crop_width), preserve_aspect_ratio=True)
        return tf.image.resize_with_crop_or_pad(image, self.height, self.width)


class ImageRandomCropTransformation(CropTransformation):

    def transform(self, image: tf.Tensor) -> tf.Tensor:
        crop_height, crop_width = [
            int(x * self.crop_adjustment) for x in (self.height, self.width)
        ]
        image = tf.image.resize(image, (crop_height, crop_width))
        return tf.image.random_crop(image, self.image_dimensions)
