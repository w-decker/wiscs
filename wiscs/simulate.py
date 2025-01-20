from . import config
from .params import validate_params, parse_params, update_params

import numpy as np
import warnings
import numpy.typing as npt
import pandas as pd
import copy

import numpy as np
import warnings
import numpy.typing as npt

def generate(params:dict, seed:int=None):

    if seed is not None:
        np.random.seed(seed)

    # get experimental parameters
    n_sub = params["n"]["subject"]
    n_ques = params["n"]["question"]
    n_item = params["n"]["trial"]

    w_perceptual = params["word"]["perceptual"]
    w_conceptual = params["word"]["conceptual"]
    w_task_times = params["word"]["task"]

    i_perceptual = params["image"]["perceptual"]
    i_conceptual = params["image"]["conceptual"]
    i_task_times = params["image"]["task"]

    Sigma_sub   = params["var"]["subject"]   # shape (2,2)
    Sigma_ques  = params["var"]["question"]  # shape (2,2)
    Sigma_item  = params["var"]["item"]      # shape (2,2)
    sigma_error = params["var"]["error"]     # float

    uv_sub = np.random.multivariate_normal(mean=[0, 0], cov=Sigma_sub, size=n_sub)
    sub_u = uv_sub[:, 0]  # random intercept for subject
    sub_v = uv_sub[:, 1]  # random slope

    uv_ques = np.random.multivariate_normal(mean=[0, 0], cov=Sigma_ques, size=n_ques)
    ques_u = uv_ques[:, 0]  # random intercept for question
    ques_v = uv_ques[:, 1]  # random slope

    uv_item = np.random.multivariate_normal(mean=[0, 0], cov=Sigma_item, size=n_item)
    item_u = uv_item[:, 0]  # random intercept for item
    item_v = uv_item[:, 1]  # random slope

    S_grid, Q_grid, I_grid = np.meshgrid(
        np.arange(n_sub),
        np.arange(n_ques),
        np.arange(n_item),
        indexing='ij'
    )

    word_random = (
        sub_u[S_grid]          # subject intercept
        + ques_u[Q_grid]       # question intercept
        + item_u[I_grid]       # item intercept
    )
    # plus a distinct residual error for each cell
    word_error = np.random.normal(0, sigma_error, size=(n_sub, n_ques, n_item))

    image_random = (
        (sub_u[S_grid] + sub_v[S_grid])       # subject intercept + slope
        + (ques_u[Q_grid] + ques_v[Q_grid])   # question intercept + slope
        + (item_u[I_grid] + item_v[I_grid])   # item intercept + slope
    )
    image_error = np.random.normal(0, sigma_error, size=(n_sub, n_ques, n_item))

    word_base = (
        w_perceptual
        + w_conceptual
        + w_task_times[Q_grid]  # shape (n_sub, n_ques, n_item)
    )
    image_base = (
        i_perceptual
        + i_conceptual
        + i_task_times[Q_grid]
    )

    word = word_base + word_random + word_error
    image = image_base + image_random + image_error

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

    def fit_transform(self, params:dict=None, overwrite:bool=False):
        if np.array_equal(self.params["word"]["task"], self.params["image"]["task"]):
            warnings.warn("Simulating data for MAIN hypothesis.")
        else:
            warnings.warn("Simulating data for ALTERNATIVE hypothesis.")
        
        if overwrite:
            if params is None:
                raise ValueError("If overwrite is True, params must be provided")
            elif params is not None and len(params) != len(self.params):
                self.params = update_params(self.params, params)
                self.data = generate(self.params)
            elif params is not None and len(params) == len(self.params):
                validate_params(params)
                self.params = parse_params(params)
                self.data = generate(self.params)
        else:
            if params is not None and len(params) == len(self.params):
                validate_params(params)
                self.data = generate(parse_params(params))
            elif params is not None and len(params) != len(self.params):
                params = update_params(self.params, params)
                self.data = generate(params)
            else:
                self.data = generate(self.params)

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