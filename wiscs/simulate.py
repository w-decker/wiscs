from . import config
from .params import validate_params, parse_params, update_params

import numpy as np
import warnings
import numpy.typing as npt
import pandas as pd
import copy
from typing import Union

def _random_effects(sigma:npt.ArrayLike,
                    size:Union[int, tuple],
                    mean:npt.ArrayLike = [0, 0]) -> npt.ArrayLike:
    """Generate correlated random intercepts and slopes."""
    v = np.random.multivariate_normal(mean, sigma, size)
    intercept = v[:, 0]
    slope = v[:, 1]
    return intercept, slope

def generate(params:dict, seed:int=None):

    # Number of subjects, questions, items
    n_subs = params["n"]["subject"]
    n_questions = params["n"]["question"]
    n_items = params["n"]["item"]

    # Fixed effects components (example parameters)
    w_perceptual = params["word"]["perceptual"]
    w_conceptual = params["word"]["conceptual"]
    w_task = params["word"]["task"]

    i_perceptual = params["image"]["perceptual"]
    i_conceptual = params["image"]["conceptual"]
    i_task = params["image"]["task"]

    # Variance components
    sd_question = params["sd"]["question"]
    sd_item = params["sd"]["item"] # could take None, int, float, np.ndarray [2x2 var-cov matrix]
    sd_subject = params["sd"]["subject"]  # could take None, int, float, np.ndarray [2x2 var-cov matrix]
    error = params["sd"]["error"]

    if seed is not None:
        np.random.seed(seed)

    # Random effects for subjects: random intercept & slope for question
    if isinstance(sd_subject, (int, float)):
        q_intercept = np.random.normal(0, sd_subject, n_subs)
        q_slope = np.zeros(n_subs)
    elif sd_subject is None:
        q_intercept = np.zeros(n_subs)
        q_slope = np.zeros(n_subs)
    else:
        q_intercept, q_slope = _random_effects(sd_subject, n_subs)

    # Random intercepts and slopes for items
    if isinstance(sd_item, (int, float)):
        item_effects = np.random.normal(0, sd_item, n_items)
        i_slope = np.zeros(n_items)
    elif sd_item is None:
        item_effects = np.zeros(n_items)
        i_slope = np.zeros(n_items)
    elif isinstance(sd_item, np.ndarray) and sd_item.shape == (2, 2):
        item_effects, i_slope = _random_effects(sd_item, n_items)
    else:
        raise ValueError("sd_item must be None, a scalar, or a 2x2 covariance matrix.")

    # Random intercepts for question
    question_effects = np.random.normal(0, sd_question, n_questions) if sd_question is not None else np.zeros(n_questions)

    # Fixed RT without noise
    fixed_rt_word = w_perceptual + w_conceptual + w_task
    fixed_rt_image = i_perceptual + i_conceptual + i_task

    # Reshape for broadcasting
    q_intercept = q_intercept[:, np.newaxis, np.newaxis]  # (n_subs, 1, 1)
    q_slope = q_slope[:, np.newaxis, np.newaxis]  # (n_subs, 1, 1)
    fixed_rt_word = fixed_rt_word[np.newaxis, :, np.newaxis]  # (1, n_questions, 1)
    fixed_rt_image = fixed_rt_image[np.newaxis, :, np.newaxis]  # (1, n_questions, 1)
    question_effects = question_effects[np.newaxis, :, np.newaxis]  # (1, n_questions, 1)
    item_effects = item_effects[np.newaxis, np.newaxis, :]  # (1, 1, n_items)

    # Apply slope effects
    q_indices = np.arange(n_questions) / n_questions
    q_indices = q_indices.reshape(1, n_questions, 1)  # Shape: (1, n_questions, 1)
    q_slope = q_slope * q_indices  # Random slope for question

    i_indices = np.arange(n_items) / n_items
    i_indices = i_indices.reshape(1, 1, n_items)  # Shape: (1, 1, n_items)
    i_slope = i_slope * i_indices  # Random slope for modality by item

    # Residual error
    residual_word = np.random.normal(0, error, size=(n_subs, n_questions, n_items)) if error is not None else np.zeros((n_subs, n_questions, n_items))
    residual_image = np.random.normal(0, error, size=(n_subs, n_questions, n_items)) if error is not None else np.zeros((n_subs, n_questions, n_items))

    # Generate word and image data
    word = (
        fixed_rt_word
        + q_intercept
        + q_slope
        + i_slope
        + question_effects
        + item_effects
        + residual_word
    )

    image = (
        fixed_rt_image
        + q_intercept
        + q_slope
        + i_slope
        + question_effects
        + item_effects
        + residual_image
    )

    return word, image

class DataGenerator(object):
    """Data generator
    
    Methods
    -------
    fit_transform(self, params:dict=None, overwrite:bool=False)
        Generate data based on parameters

    to_pandas(self) -> pd.DataFrame
        Convert data to pandas dataframe

    Attributes
    ----------
    data: npt.ArrayLike | npt.ArrayLike
        Generated data (image, word)

    word: npt.ArrayLike
        Generated word data

    image: npt.ArrayLike
        Generated image data
    """ 

    def __init__(self):
        self.params = copy.deepcopy(config.p)

    def fit_transform(self, params:dict=None, overwrite:bool=False, seed:int=None):
        if np.array_equal(self.params["word"]["task"], self.params["image"]["task"]):
            warnings.warn("Simulating data for MAIN hypothesis.")
        else:
            warnings.warn("Simulating data for ALTERNATIVE hypothesis.")
        
        if overwrite:
            if params is None:
                raise ValueError("If overwrite is True, params must be provided")
            elif params is not None and len(params) != len(self.params):
                self.params = update_params(self.params, params)
                self.data = generate(self.params, seed=seed)
                self.word = self.data[0]
                self.image = self.data[1]
            elif params is not None and len(params) == len(self.params):
                validate_params(params)
                self.params = parse_params(params)
                self.data = generate(self.params, seed=seed)
                self.word = self.data[0]
                self.image = self.data[1]
        else:
            if params is not None and len(params) == len(self.params):
                validate_params(params)
                self.data = generate(parse_params(params), seed=seed)
            elif params is not None and len(params) != len(self.params):
                params = update_params(self.params, params)
                self.data = generate(params, seed=seed)
                self.word = self.data[0]
                self.image = self.data[1]
            else:
                self.data = generate(self.params, seed=seed)
                self.word = self.data[0]
                self.image = self.data[1]
        return self   
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame.
        """
        # word
        n_participants, n_questions, n_items = self.word.shape
        word_df = pd.DataFrame({
            "subject": np.repeat(np.arange(n_participants), n_questions * n_items),
            "rt": self.data[0].flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_items), n_participants),
            "item": np.tile(np.arange(n_items), n_participants * n_questions),
            "modality": "image"
        })

        # image
        n_participants, n_questions, n_items = self.image.shape
        image_df = pd.DataFrame({
            "subject": np.repeat(np.arange(self.data[1].shape[0]), n_questions * n_items),
            "rt": self.data[1].flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_items), self.data[1].shape[0]),
            "item": np.tile(np.arange(n_items), n_participants * n_questions),
            "modality": "word"
        })

        # concatenate
        df = pd.concat([image_df, word_df], ignore_index=True)
        return df