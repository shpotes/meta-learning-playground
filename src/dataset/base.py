"""
Provide some base methods and classes from different types of datasets
"""

from pathlib import PosixPath, Path
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Union

class MetaDataset:
    """
    A meta-dataset, just a set of dataset

    ...

    Attributes
    ----------
    base_dataste : str
        The name of the dataset from which the episodes will be extracted.
    input_shape : Tuple[int, int]
        Image shape
    n_classes : int
        Number of classes per episode. None if the number of classes isn't fixed
    k_shots : int
        Number of examples per class per episode. None if the set isn't perfectly balanced
    q_size : int
        Number of examples per query per episode. None if the query size isn't constant
    is_episodical : bool
        Whether the set is episodical or not
    metadata : pd.DataFrame
        It contains all the relevant information of each point

    Methods
    -------
    sources : List[Tuples]

    load : tf.Tensor -> tf.Tensor

    dataset : tf.data.Dataset
    """
    def __init__(self,
                 base_dataset: str,
                 input_shape: Tuple[int, int],
                 n_classes: int,
                 k_shots: int,
                 q_size: int,
                 metadata: pd.DataFrame):
        self.name = f'{self.__class__.name__}_{base_dataset}'
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.k_shots = self.k_shots
        self.q_size = self.q_size
        self.metadata = metadata

        self.is_episodical = None in (n_classes, k_shots)

    def _sources(self) -> List[Tuples]:
        return [()]

    def load(fname: Union[str, tf.Tensor]) -> tf.Tensor:
        """
        Load image tensor

        Parameters
        ----------
        fname
            file name to be loaded

        Returns
        -------
        img
            The corresponding loaded image
        """
        return fname

    def preprocess(raw: tf.Tensor) -> tf.Tensor:
        """
        Normalize image range

        Parameters
        ----------
        raw
            The input raw image to be preprocessed

        Returns
        -------
        img
            The corresponding processed image
        """
        img = tf.cast(raw, tf.float32)
        img = tf.image.resize(img, self.input_shape)
        img = img - tf.reduce_min(img)
        img = img / tf.reduce_max(img)
        return img

    def dataset(self) -> tf.data.Dataset:
        pass


def generate_metadata_from_raw_folder(raw_folder: PosixPath)-> pd.DataFrame:
    """
    Generate metadata file from a given folder.

    Parameters
    ----------
    raw_folder
        The root folder where the raw dataset was extracted

    Returns
    -------
    dataset
        Data columns are as follows:

        ===== ================
        TODO: define structure
    """
    pass
