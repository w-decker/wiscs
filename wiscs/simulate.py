import numpy as np
from typing import Union
import copy
import warnings
import pandas as pd

from .formula import Formula
from . import config
from .params import validate_params, parse_params, update_params

def calculate_baseline(params:dict):
    """Calculate baseline based on parameters"""

    def create_matrix(rt, task):
        matrix = np.full((params["n"]["subject"], params["n"]["question"], params["n"]["item"]), rt)
        task = task[np.newaxis, :, np.newaxis]
        return matrix + task

    word_rt = params['word']['perceptual'] + params['word']['conceptual']
    image_rt = params['image']['perceptual'] + params['image']['conceptual']

    word_matrix = create_matrix(word_rt, params["word"]["task"])
    image_matrix = create_matrix(image_rt, params["image"]["task"])

    return np.stack((word_matrix, image_matrix), axis=3)

def generate_correlated_effects(n:int, 
                                sd_values:Union[list[int], list[float], np.ndarray], 
                                corr_matrix: Union[np.ndarray, list[list]]):
    """
    Generates correlated random effects using a multivariate normal distribution 
    and Cholesky decomposition.

    Parameters
    ----------
    n:int 
    sd_values: list
    corr_matrix: Union[np.ndarray, list[list]].

    Return
    -------
    np.ndarray: A (n, len(sd_values)) array of random effects.
    """
    # transform correlation matrix to covariance matrix
    sd_diag = np.diag(sd_values)
    cov_matrix = sd_diag @ np.array(corr_matrix) @ sd_diag  # Covariance matrix

    # RE matrix using Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    z = np.random.randn(n, len(sd_values))  # Independent standard normal samples

    return z @ L.T

def make_random_effects(params:dict):
    """
    Generates random effects strucrure based on the formula specified in the parameters.
    """
    re_formula = params["sd"].get("re_formula")
    if not re_formula:
        raise ValueError("Missing 'sd.re_formula'")

    formula = Formula(re_formula)
    random_effects = {}

    for term in formula:
        content = term.strip("()")
        lhs, group = content.split("|")
        lhs = lhs.strip()
        group = group.strip()

        n_key = f"{group}"
        if n_key not in params["n"]:
            raise ValueError(f"Missing 'n.{n_key}' for grouping factor '{group}'.")

        # get intercept and slopes
        predictors = [t.strip() for t in lhs.split("+")]
        has_intercept = "1" in predictors
        slopes = [p for p in predictors if p != "1"]

        # ordered list of effects: [intercept, slopes]
        effect_names = ["intercept"] if has_intercept else []
        effect_names.extend(slopes)

        # get sd
        sd_values = [params["sd"].get(group, 0) if has_intercept else 0]
        sd_values.extend([params["sd"].get(k, 0) for k in slopes])

        # get correlations
        corr_matrix = np.eye(len(sd_values))
        if group in params["corr"] and len(params["corr"][group]) == len(sd_values):
            corr_matrix = np.array(params["corr"][group])

        # create random effects
        random_effects[group] = generate_correlated_effects(params["n"][n_key], sd_values, corr_matrix)

    return random_effects

def generate(params:dict, seed:int=None):
    """
    Generate data
    
    Returns
    -------
    np.ndarray: A (subjects, questions, items, modalities) matrix containing simulated RT values.
    """
    if seed is not None:
        np.random.seed(seed)

    B = calculate_baseline(params)

    # create random effects
    random_effects = make_random_effects(params)

    n_subj, n_q, n_item, n_mod = B.shape 

    # get fixed effects
    modality_code = np.array([0, 1]).reshape(1, 1, 1, n_mod)  # 0 for word, 1 for image
    question_code = (np.arange(n_q) - np.mean(np.arange(n_q))).reshape(1, n_q, 1, 1)  # Centered

    total_contrib = np.zeros_like(B)

    # insert random effects 
    formula = Formula(params["sd"]["re_formula"])  # Parse the formula
    for term in formula:

        # Extract structure: (effects | grouping factor)
        content = term.strip("()")
        lhs, group = content.split("|")
        lhs = lhs.strip()
        group = group.strip()

        # Ensure we have the number of levels for this grouping factor
        n_levels = params["n"].get(f"{group}", None)
        if n_levels is None:
            raise ValueError(f"Missing `n.{group}` for grouping factor '{group}'.")

        # Get the random effect matrix for this factor
        re_matrix = random_effects.get(group, None)
        if re_matrix is None:
            continue  # Skip if no random effects for this factor

        effect_names = lhs.split("+")
        effect_names = [e.strip() for e in effect_names]

        # Initialize contribution per factor
        factor_contrib = np.zeros_like(B)

        # Broadcast effects
        effect_idx = 0
        for term in effect_names:
            if term == "1":
                # Random intercept
                if group == "subject":
                    factor_contrib += re_matrix[:, effect_idx].reshape(n_levels, 1, 1, 1)
                elif group == "question":
                    factor_contrib += re_matrix[:, effect_idx].reshape(1, n_levels, 1, 1)
                elif group == "item":
                    factor_contrib += re_matrix[:, effect_idx].reshape(1, 1, n_levels, 1)
            elif term == "modality":
                if group == "subject":
                    factor_contrib += re_matrix[:, effect_idx].reshape(n_levels, 1, 1, 1) * modality_code
                elif group == "question":
                    factor_contrib += re_matrix[:, effect_idx].reshape(1, n_levels, 1, 1) * modality_code
                elif group == "item":
                    factor_contrib += re_matrix[:, effect_idx].reshape(1, 1, n_levels, 1) * modality_code
            elif term == "question":
                if group == "subject":
                    factor_contrib += re_matrix[:, effect_idx].reshape(n_levels, 1, 1, 1) * question_code
                elif group == "question":
                    factor_contrib += re_matrix[:, effect_idx].reshape(1, n_levels, 1, 1) * question_code
                elif group == "item":
                    factor_contrib += re_matrix[:, effect_idx].reshape(1, 1, n_levels, 1) * question_code
            effect_idx += 1

        # Add contribution to total
        total_contrib += factor_contrib

    # add residual
    residual_noise = np.random.normal(0, params["sd"]["error"], size=B.shape)

    return B + total_contrib + residual_noise

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
                self.word = self.data[:, :, :, 0]
                self.image = self.data[:, :, :, 1]
            elif params is not None and len(params) == len(self.params):
                validate_params(params)
                self.params = parse_params(params)
                self.data = generate(self.params, seed=seed)
                self.word = self.data[:, :, :, 0]
                self.image = self.data[:, :, :, 1]
        else:
            if params is not None and len(params) == len(self.params):
                validate_params(params)
                self.data = generate(parse_params(params), seed=seed)
            elif params is not None and len(params) != len(self.params):
                params = update_params(self.params, params)
                self.data = generate(params, seed=seed)
                self.word = self.data[:, :, :, 0]
                self.image = self.data[:, :, :, 1]
            else:
                self.data = generate(self.params, seed=seed)
                self.word = self.data[:, :, :, 0]
                self.image = self.data[:, :, :, 1]
        return self   
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame.
        """
        # word
        n_participants, n_questions, n_items = self.word.shape
        word_df = pd.DataFrame({
            "subject": np.repeat(np.arange(n_participants), n_questions * n_items),
            "rt": self.word.flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_items), n_participants),
            "item": np.tile(np.arange(n_items), n_participants * n_questions),
            "modality": "word"
        })

        # image
        n_participants, n_questions, n_items = self.image.shape
        image_df = pd.DataFrame({
            "subject": np.repeat(np.arange(self.image.shape[0]), n_questions * n_items),
            "rt": self.image.flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_items), n_participants),
            "item": np.tile(np.arange(n_items), n_participants * n_questions),
            "modality": "image"
        })

        # concatenate
        df = pd.concat([image_df, word_df], ignore_index=True)
        return df