from . import config
from .params import validate_params, parse_params, update_params

import numpy as np
import warnings
import numpy.typing as npt
import pandas as pd
import copy

def make_tasks(low, high, n) -> npt.ArrayLike:
    """Generate task parameters"""
    return np.random.permutation(np.linspace(low, high, n).round(0))

def generate(params:dict) -> npt.ArrayLike | npt.ArrayLike:
    """Generate data
    
    Returns
    -------
    image, word
    """

    if not np.array_equal(params["image"]["task"], params["word"]["task"]):
        warnings.warn("Tasks parameters are different. Generating data for ALTERNATIVE hypothesis.")

    additional_image_vars = sum(
        params["image"].get(key, 0) 
        for key in params["image"] if key not in ["concept", "task"]
    )

    additional_word_vars = sum(
        params["word"].get(key, 0) 
        for key in params["word"] if key not in ["concept", "task"]
    )

    # noise distributions
    var_item_image = np.random.normal(0, params["var"]["image"], (params["n"]["participant"], params["n"]["question"], params["n"]["trial"]))
    var_item_word = np.random.normal(0, params["var"]["word"], (params["n"]["participant"], params["n"]["question"], params["n"]["trial"]))
    var_question = np.random.normal(0, params["var"]["question"], (params["n"]["participant"], params["n"]["question"], params["n"]["trial"]))
    var_participant = np.random.normal(0, params["var"]["participant"], (params["n"]["participant"], params["n"]["question"], params["n"]["trial"]))

    return \
            (params["image"]["concept"]
            + additional_image_vars
           + params["image"]["task"][None, :, None] 
           + var_item_image
           + var_participant
           + var_question), \
            (params["word"]["concept"]
             + additional_word_vars
            + params["word"]["task"][None, :, None]
            + var_item_word
            + var_question
            + var_participant)

class DataGenerator(object):
    """Data generator
    
    Methods
    -------
    fit(self, params:dict=None, overwrite:bool=False)
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

    def fit(self, params:dict=None, overwrite:bool=False):
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
        """Convert data to pandas dataframe"""
        
        df_image = pd.DataFrame({
            'participant': np.repeat(np.arange(self.params["n"]["participant"]), self.params["n"]["question"] * self.params["n"]["trial"]),
            'question': np.tile(np.repeat(np.arange( self.params["n"]["question"]), self.params["n"]["trial"]), self.params["n"]["participant"]),
            'trial': np.tile(np.arange(self.params["n"]["trial"]), self.params["n"]["participant"] * self.params["n"]["question"]),
            'RT': self.data[0].flatten() # image data
        })

        df_word = pd.DataFrame({
            'participant': np.repeat(np.arange(self.params["n"]["participant"]), self.params["n"]["question"] * self.params["n"]["trial"]),
            'question': np.tile(np.repeat(np.arange( self.params["n"]["question"]), self.params["n"]["trial"]), self.params["n"]["participant"]),
            'trial': np.tile(np.arange(self.params["n"]["trial"]), self.params["n"]["participant"] * self.params["n"]["question"]),
            'RT': self.data[1].flatten() # word data
        })

        df_image['modality'] = 'image'
        df_word['modality'] = 'word'
        return pd.concat([df_image, df_word], ignore_index=True)