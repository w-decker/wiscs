from . import config
from .params import validate_params, parse_params, update_params

import numpy as np
import warnings
import numpy.typing as npt
import pandas as pd
import copy
from typing import Union

import numpy as np
import warnings
import numpy.typing as npt

MODALITIES = ["word", "image"]

def _random_effects(sigma:npt.ArrayLike, size:Union[int, tuple], mean:npt.ArrayLike=[0, 0]) -> npt.ArrayLike:
    """Generate random effects"""
    v = np.random.multivariate_normal(mean, sigma, size)
    intercept = v[:, 0]
    slope = v[:, 1]
    return intercept, slope

def generate(params:dict, seed:int=None):

    if seed is not None:
        np.random.seed(seed)

    n_subs = params["n"]["subject"]
    n_questions = params["n"]["question"]
    n_items = params["n"]["trial"]

    w_perceptual = params["word"]["perceptual"]
    w_conceptual = params["word"]["conceptual"]
    w_task = params["word"]["task"]

    i_perceptual = params["image"]["perceptual"]
    i_conceptual = params["image"]["conceptual"]
    i_task = params["image"]["task"]

    # variance components
    sd_question = params["var"]["question"]
    sd_item = params["var"]["trial"]
    cov_subject = params["var"]["subject"] # 2 x 2 covariance matrix for subject
    error = params["var"]["error"]

    # Seed for reproducibility
    np.random.seed(123)

    # generate variance structures
    beta0, beta1 = _random_effects(cov_subject, n_subs) # random slope and intercept for subjects
    question_effects = np.random.normal(0, sd_question, n_questions)
    item_effects = np.random.normal(0, sd_item, n_items)

    # fixed RT without noise
    fixed_rt_word = w_perceptual + w_conceptual + w_task
    fixed_rt_image = i_perceptual + i_conceptual + i_task

    # reshape
    beta0 = beta0[:, np.newaxis, np.newaxis]
    beta1 = beta1[:, np.newaxis, np.newaxis]
    fixed_rt_word = fixed_rt_word[np.newaxis, :, np.newaxis]
    fixed_rt_image = fixed_rt_image[np.newaxis, :, np.newaxis]
    question_effects = question_effects[np.newaxis, :, np.newaxis]
    item_effects = item_effects[np.newaxis, np.newaxis, :]
    q_indices = np.arange(1, n_questions + 1).reshape(1, n_questions, 1)

    # slope
    slope_q = beta1 * q_indices

    # residual/error
    residual_word = np.random.normal(0, error, size=(n_subs, n_questions, n_items))
    residual_image = np.random.normal(0, error, size=(n_subs, n_questions, n_items))

    word = (
        fixed_rt_word          # Fixed components for WORD
        + beta0                # Subject random intercept
        + slope_q              # Subject random slope effect
        + question_effects     # Question random effects
        + item_effects         # Item random effects
        + residual_word        # Residual noise
    )

    image = (
        fixed_rt_image         # Fixed components for IMAGE
        + beta0                # Subject random intercept
        + slope_q              # Subject random slope effect
        + question_effects     # Question random effects
        + item_effects         # Item random effects
        + residual_image       # Residual noise
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
            elif params is not None and len(params) == len(self.params):
                validate_params(params)
                self.params = parse_params(params)
                self.data = generate(self.params, seed=seed)
        else:
            if params is not None and len(params) == len(self.params):
                validate_params(params)
                self.data = generate(parse_params(params), seed=seed)
            elif params is not None and len(params) != len(self.params):
                params = update_params(self.params, params)
                self.data = generate(params, seed=seed)
            else:
                self.data = generate(self.params, seed=seed)

        return self    
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame.
        """
        # image
        n_participants, n_questions, n_trials = self.data[0].shape
        word_df = pd.DataFrame({
            "subject": np.repeat(np.arange(n_participants), n_questions * n_trials),
            "rt": self.data[0].flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_trials), n_participants),
            "item": np.tile(np.arange(n_trials), n_participants * n_questions),
            "modality": "image"
        })

        # word
        image_df = pd.DataFrame({
            "subject": np.repeat(np.arange(self.data[1].shape[0]), n_questions * n_trials),
            "rt": self.data[1].flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_trials), self.data[1].shape[0]),
            "item": np.tile(np.arange(n_trials), n_participants * n_questions),
            "modality": "word"
        })

        # concatenate
        df = pd.concat([image_df, word_df], ignore_index=True)
        return df