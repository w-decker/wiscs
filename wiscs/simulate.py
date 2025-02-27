import numpy as np
from typing import Union
import copy
import warnings
import pandas as pd

from .formula import Formula
from . import config
from .params import validate_params, parse_params, update_params

def calculate_baseline(params: dict):
    """Create a baseline matrix"""
    def create_matrix(rt, task):

        matrix = np.full(
            (params["n"]["subject"], params["n"]["question"], params["n"]["item"]), 
            float(rt),
            dtype=float
        )
        task = task.astype(float)[np.newaxis, :, np.newaxis]
        return matrix + task

    word_rt = params['word']['perceptual'] + params['word']['conceptual']
    image_rt = params['image']['perceptual'] + params['image']['conceptual']

    word_matrix = create_matrix(word_rt, params["word"]["task"])
    image_matrix = create_matrix(image_rt, params["image"]["task"])

    return np.stack((word_matrix, image_matrix), axis=3).astype(float)


def generate_correlated_effects(n: int,
                                sd_values: Union[list[int], list[float], np.ndarray],
                                corr_matrix: Union[np.ndarray, list[list]],
                                method: str = "cholesky"):
    """
    Generates correlated random effects using a multivariate normal distribution 
    and Cholesky decomposition or eigen-value decomposition.

    Parameters
    ----------
    n:int 
    sd_values: list
    corr_matrix: Union[np.ndarray, list[list]].
    method: str, default="cholesky"
    
    Return
    -------
    np.ndarray: A (n, len(sd_values)) array of random effects.
    """
    sd_values = np.asarray(sd_values, dtype=float)
    k = len(sd_values)
    if sd_values.ndim != 1:
        raise ValueError("sd_values must be 1D")

    # construct variance-covariance matrix
    cov_matrix = np.diag(sd_values) @ np.array(corr_matrix) @ np.diag(sd_values)

    # decompose with Cholesky or eigen-value decomposition
    if method == "cholesky":
        L = np.linalg.cholesky(cov_matrix)
    elif method == "eigen":
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals[eigvals < 1e-12] = 0 # zero out negative eigenvals
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    else:
        raise ValueError("Unknown method. Must be 'cholesky' or 'eigen'.")

    z = np.random.randn(n, k)  # independent normal
    return z @ L.T

def _get_question_codes(params: dict) -> np.ndarray:
    """
    Returns a 2D design matrix for the 'question' factor/predictor.

    If 'sd.question' is a single number (float), we treat 'question' as numeric 
    with Q levels, coded e.g. [-2, -1, 0, +1, +2] (mean-centered).
    That yields shape (Q,1).

    If 'sd.question' is a list of length (Q-1), we treat 'question' as factor-coded,
    i.e. one-hot or treatment coding with Q-1 dummy columns, shape (Q, Q-1).

    If 'sd.question' is not present, or set to None, we return shape (Q, 0) 
    which means no slope for question.
    """
    Q = params["n"]["question"]
    sdval = params["sd"].get("question", None)
    if sdval is None:
        # user didn't specify a random slope for question
        return np.zeros((Q, 0), dtype=float)

    if isinstance(sdval, (float, int)):
        x = np.arange(Q, dtype=float)
        x -= x.mean()  # mean-center
        return x.reshape(Q, 1)

    if isinstance(sdval, list):
        if len(sdval) != (Q - 1):
            raise ValueError(
                f"For factor-coded 'question' with n.question={Q}, "
                f"sd.question must be length {Q-1}, but got {len(sdval)}."
            )
        codes = np.zeros((Q, Q - 1), dtype=float)
        for lev in range(1, Q):
            codes[lev, lev - 1] = 1.0
        return codes

    raise TypeError("sd.question must be float/int or a list of length Q-1 if factor-coded.")

def make_random_effects(params: dict):
    re_formula = params["sd"].get("re_formula")
    if not re_formula:
        raise ValueError("Missing 'sd.re_formula' in params['sd'].")

    formula = Formula(re_formula)
    random_effects = {}

    question_mat = _get_question_codes(params)
    n_question_slopes = question_mat.shape[1]

    for term in formula:
        # e.g. '(1 + question | subject)'
        content = term.strip("()")
        lhs, group = content.split("|")
        lhs = lhs.strip()   # '1 + question'
        group = group.strip()  # 'subject'

        n_key = f"{group}"
        if n_key not in params["n"]:
            raise ValueError(f"Missing 'n.{group}' for grouping factor '{group}'.")

        predictors = [t.strip() for t in lhs.split("+")]  # e.g. ['1', 'question']
        has_intercept = ("1" in predictors)
        slopes = [p for p in predictors if p != "1"]

        # random intercept SD for that group
        intercept_sd_val = params["sd"].get(group, 0)
        intercept_sd = [float(intercept_sd_val)] if has_intercept else []

        # gather slope sds
        slopes_sd = []
        for p in slopes:
            if p == "question":
                sd_val = params["sd"].get("question", 0)
                if isinstance(sd_val, list):
                    slopes_sd.extend(float(x) for x in sd_val)
                else:
                    slopes_sd.append(float(sd_val))
            elif p == "modality":
                sd_value = params["sd"].get("modality", 0)
                if isinstance(sd_value, list):
                    slopes_sd.extend([float(v) for v in sd_value])
                else:
                    slopes_sd.append(float(sd_value))
            else:
                sd_value = params["sd"].get(p, 0)
                if isinstance(sd_value, list):
                    slopes_sd.extend([float(v) for v in sd_value])
                else:
                    slopes_sd.append(float(sd_value))

        # total param dimension
        sd_values = np.array(intercept_sd + slopes_sd, dtype=float)
        k_params = len(sd_values)

        # correlation matrix for group
        corr_matrix = np.eye(k_params)
        if group in params["corr"] and params["corr"][group] is not None:
            user_corr = np.array(params["corr"][group])
            if user_corr.shape != (k_params, k_params):
                raise ValueError(
                    f"{group}: correlation matrix must be {k_params}x{k_params}, "
                    f"but got {user_corr.shape}."
                )
            corr_matrix = user_corr

        # generate random effects for that group
        re_matrix = generate_correlated_effects(
            params["n"][n_key],
            sd_values,
            corr_matrix,
            method=config.re_method
        )

        random_effects[group] = re_matrix  # shape (n_levels, k_params)

    return random_effects

def generate(params: dict, seed: int = None):
    """
    Generate data
    
    Returns
    -------
    np.ndarray: A (subjects, questions, items, modalities) matrix containing simulated RT values.
    """
    if seed is not None:
        np.random.seed(seed)

    # fixed baseline
    B = calculate_baseline(params)

    # random effects and makeing some codes
    random_effects = make_random_effects(params)
    n_subj, n_q, n_item, n_mod = B.shape
    total_contrib = np.zeros_like(B)
    question_codes = _get_question_codes(params)
    m_question = question_codes.shape[1]
    question_codes_5d = question_codes.reshape(1, n_q, 1, 1, m_question)
    modality_code = np.array([0, 1]).reshape(1, 1, 1, n_mod)

    # parse formula
    formula = Formula(params["sd"]["re_formula"])

    for term in formula:
        content = term.strip("()")
        lhs, group = content.split("|")
        lhs = lhs.strip()   # e.g. '1 + question'
        group = group.strip()

        # get the random effect matrix for this group
        re_matrix = random_effects.get(group, None)
        if re_matrix is None:
            continue

        effect_names = lhs.split("+")
        effect_names = [e.strip() for e in effect_names]
        factor_contrib = np.zeros_like(B)

        # track which column of re_matrix we're using
        next_col = 0

        for p in effect_names:
            if p == "1":
                # random intercept
                if group == "subject":
                    factor_contrib += re_matrix[:, next_col].reshape(n_subj, 1, 1, 1)
                elif group == "question":
                    factor_contrib += re_matrix[:, next_col].reshape(1, n_q, 1, 1)
                elif group == "item":
                    factor_contrib += re_matrix[:, next_col].reshape(1, 1, n_item, 1)
                elif group == "modality":
                    factor_contrib += re_matrix[:, next_col].reshape(1, 1, 1, n_mod)
                next_col += 1

            elif p == "question":
                remain = re_matrix.shape[1] - next_col
                num_slopes = min(remain, m_question)
                for sc in range(num_slopes):
                    slope_vals = re_matrix[:, next_col + sc]
                    if group == "subject":
                        slope_4d = slope_vals.reshape(n_subj, 1, 1, 1)
                        code_4d = question_codes_5d[0, :, 0, 0, sc].reshape(1, n_q, 1, 1)
                        factor_contrib += slope_4d * code_4d

                    elif group == "question":
                        slope_4d = slope_vals.reshape(1, n_q, 1, 1)
                        factor_contrib += slope_4d * question_codes_5d[0,:,:,:,sc]

                    elif group == "item":
                        slope_4d = slope_vals.reshape(1, 1, n_item, 1)
                        code_4d = question_codes_5d[0, :, 0, 0, sc].reshape(1, n_q, 1, 1)
                        factor_contrib += slope_4d * code_4d

                    elif group == "modality":
                        slope_4d = slope_vals.reshape(1, 1, 1, n_mod)
                        code_4d = question_codes_5d[0, :, 0, 0, sc].reshape(1, n_q, 1, 1)
                        factor_contrib += slope_4d * code_4d

                next_col += num_slopes

            elif p == "modality":
                slope_vals = re_matrix[:, next_col]
                if group == "subject":
                    slope_4d = slope_vals.reshape(n_subj, 1, 1, 1)
                    factor_contrib += slope_4d * modality_code  # broadcast
                elif group == "question":
                    slope_4d = slope_vals.reshape(1, n_q, 1, 1)
                    factor_contrib += slope_4d * modality_code
                elif group == "item":
                    slope_4d = slope_vals.reshape(1, 1, n_item, 1)
                    factor_contrib += slope_4d * modality_code
                elif group == "modality":
                    factor_contrib += slope_vals.reshape(1, 1, 1, n_mod) * modality_code
                next_col += 1

            else:
                slope_vals = re_matrix[:, next_col]
                # broadcast it depending on grouping factor
                if group == "subject":
                    factor_contrib += slope_vals.reshape(n_subj, 1, 1, 1)
                elif group == "question":
                    factor_contrib += slope_vals.reshape(1, n_q, 1, 1)
                elif group == "item":
                    factor_contrib += slope_vals.reshape(1, 1, n_item, 1)
                elif group == "modality":
                    factor_contrib += slope_vals.reshape(1, 1, 1, n_mod)
                next_col += 1

        total_contrib += factor_contrib

    # add residual noise
    if params["sd"]["error"] is not None:
        residual_noise = np.random.normal(0, params["sd"]["error"], size=B.shape)
    else:
        residual_noise = np.zeros_like(B)

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
        self.data = None
        self.word = None
        self.image = None

    def fit_transform(self, params: dict = None, overwrite: bool = False, seed: int = None):
        """
        Generate data based on the current (or new) params. 
        If overwrite=True, we replace self.params with new ones.
        If not, we only partially update them or keep them as is.
        """
        if np.array_equal(self.params["word"]["task"], self.params["image"]["task"]):
            warnings.warn("Simulating data for MAIN hypothesis.")
        else:
            warnings.warn("Simulating data for ALTERNATIVE hypothesis.")

        if overwrite:
            if params is None:
                raise ValueError("If overwrite=True, params must be provided")
            elif params is not None and len(params) != len(self.params):
                # partial updates
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
                gen_params = parse_params(params)
                self.data = generate(gen_params, seed=seed)
            elif params is not None and len(params) != len(self.params):
                # partial update
                updated = update_params(self.params, params)
                self.data = generate(updated, seed=seed)
                self.word = self.data[:, :, :, 0]
                self.image = self.data[:, :, :, 1]
            else:
                # use the existing self.params
                self.data = generate(self.params, seed=seed)
                self.word = self.data[:, :, :, 0]
                self.image = self.data[:, :, :, 1]
        return self

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert data => long-format DataFrame with columns:
        [subject, rt, question, item, modality].
        """
        if self.data is None:
            raise ValueError("No data generated yet. Call fit_transform first.")

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

        df = pd.concat([image_df, word_df], ignore_index=True)
        return df
