from image import ImageDataset


class DataContainer:
    def __init__(self, train: ImageDataset, validation: ImageDataset, test: ImageDataset) -> None:
        """
        Data container that contains ImageDataset fol model training, validation and inference
        """
        self.train = train
        self.validation = validation
        self.test = test
