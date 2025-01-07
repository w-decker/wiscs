from . import config
from .params import validate_params, parse_params, update_params

import numpy as np
import warnings
import numpy.typing as npt
import pandas as pd
import copy

def generate(params: dict) -> tuple[npt.ArrayLike | None, npt.ArrayLike | None]:
    """
    Generate data for a within- or between-subjects design.

    Parameters
    ----------
    params : dict
        Experimental parameters.
    design : str
        Either "within" (default) or "between" to specify the experimental design.

    Returns
    -------
    tuple[npt.ArrayLike | None, npt.ArrayLike | None]
        Generated data for images and words.
    """

    if not np.array_equal(params["image"]["task"], params["word"]["task"]):
        warnings.warn("Tasks parameters are different. Generating data for ALTERNATIVE hypothesis.")
    else:
        warnings.warn("Tasks parameters are the same. Generating data for MAIN hypothesis.")

    additional_image_vars = sum(
        params["image"].get(key, 0)
        for key in params["image"] if key not in ["concept", "task"]
    )

    additional_word_vars = sum(
        params["word"].get(key, 0)
        for key in params["word"] if key not in ["concept", "task"]
    )

    n_participants = params["n"]["participant"]

    # Handle within-subjects design
    if list(params['design'].values())[0] == "within":
        var_item_image = np.random.normal(0, params["var"]["image"], (n_participants, params["n"]["question"], params["n"]["trial"]))
        var_item_word = np.random.normal(0, params["var"]["word"], (n_participants, params["n"]["question"], params["n"]["trial"]))
        var_question = np.random.normal(0, params["var"]["question"], (n_participants, params["n"]["question"], params["n"]["trial"]))
        var_participant = np.random.normal(0, params["var"]["participant"], (n_participants, params["n"]["question"], params["n"]["trial"]))

        image_data = (
            params["image"]["concept"]
            + additional_image_vars
            + params["image"]["task"][None, :, None]
            + var_item_image
            + var_participant
            + var_question
        )
        
        word_data = (
            params["word"]["concept"]
            + additional_word_vars
            + params["word"]["task"][None, :, None]
            + var_item_word
            + var_question
            + var_participant
        )

        return image_data, word_data

    # Handle between-subjects design
    elif list(params['design'].values())[0] == "between":
        half_participants = n_participants // 2

        # Generate data for the first half (images)
        var_item_image = np.random.normal(0, params["var"]["image"], (half_participants, params["n"]["question"], params["n"]["trial"]))
        var_question_image = np.random.normal(0, params["var"]["question"], (half_participants, params["n"]["question"], params["n"]["trial"]))
        var_participant_image = np.random.normal(0, params["var"]["participant"], (half_participants, params["n"]["question"], params["n"]["trial"]))

        image_data = (
            params["image"]["concept"]
            + additional_image_vars
            + params["image"]["task"][None, :, None]
            + var_item_image
            + var_participant_image
            + var_question_image
        )

        # Generate data for the second half (words)
        var_item_word = np.random.normal(0, params["var"]["word"], (half_participants, params["n"]["question"], params["n"]["trial"]))
        var_question_word = np.random.normal(0, params["var"]["question"], (half_participants, params["n"]["question"], params["n"]["trial"]))
        var_participant_word = np.random.normal(0, params["var"]["participant"], (half_participants, params["n"]["question"], params["n"]["trial"]))

        word_data = (
            params["word"]["concept"]
            + additional_word_vars
            + params["word"]["task"][None, :, None]
            + var_item_word
            + var_question_word
            + var_participant_word
        )

        return image_data, word_data


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

    def fit_transform(self, params:dict=None, overwrite:bool=False):
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
        image_df = pd.DataFrame({
            "subject": np.repeat(np.arange(n_participants), n_questions * n_trials),
            "rt": self.data[0].flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_trials), n_participants),
            "modality": "image"
        })

        # word
        word_df = pd.DataFrame({
            "subject": np.repeat(np.arange(self.data[1].shape[0]), n_questions * n_trials),
            "rt": self.data[1].flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_trials), self.data[1].shape[0]),
            "modality": "word"
        })

        # concatenate
        df = pd.concat([image_df, word_df], ignore_index=True)
        return df