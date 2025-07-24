"""
GENERALIZED LINEAR MIXED EFFECTS MODEL (GLMM) SIMULATION FOR REACTION TIME (RT) DATA
"""

import numpy as np # type: ignore
from typing import Union, NamedTuple, Callable
import copy
import warnings
import pandas as pd # type: ignore
from scipy import stats # type: ignore

from .formula import Formula
from . import config
from .params import (
    validate_params, parse_params, update_params,
    get_glmm_defaults, merge_glmm_defaults,
    VALID_FAMILIES, VALID_LINKS, VALID_FAMILY_LINK_COMBINATIONS,
    DEFAULT_FAMILY_PARAMS, RT_FAMILY_CONFIGS,
    _validate_glmm_params
)
from .glm import (
    LinkFunction, DistributionFamily,
    IdentityLink, LogLink, InverseLink, SqrtLink,
    GaussianFamily, GammaFamily, InverseGaussianFamily, LogNormalFamily,
    get_link_function, get_family, validate_family_link_combination
)

def calculate_baseline(params: dict):
    """
    Create a baseline matrix on the linear predictor scale.
    
    For GLMM, this creates the linear predictor before applying 
    the inverse link function and distribution family.
    """
    def create_matrix(rt, task):
        matrix = np.full(
            (params["n"]["subject"], params["n"]["question"], params["n"]["item"]), 
            float(rt),
            dtype=float
        )
        task = np.asarray(task, dtype=float)[np.newaxis, :, np.newaxis]
        return matrix + task

    family_name = params.get('family', 'gaussian')
    link_name = params.get('link', 'identity')
    validate_family_link_combination(family_name, link_name)
    link = get_link_function(link_name)

    # baseline RTs
    word_rt = params['word']['perceptual'] + params['word']['conceptual']
    image_rt = params['image']['perceptual'] + params['image']['conceptual']
    word_matrix = create_matrix(word_rt, params["word"]["task"])
    image_matrix = create_matrix(image_rt, params["image"]["task"])
    
    # stack
    response_scale = np.stack((word_matrix, image_matrix), axis=3).astype(float)
    
    # transform via link function
    linear_predictor = link.link(response_scale)
    
    return linear_predictor

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

def get_factor_codes(factor_name: str, params: dict, coding_scheme: str = "treatment") -> np.ndarray:
    """
    Returns a 2D design matrix for any factor/predictor.

    Parameters
    ----------
    factor_name : str
        Name of the factor (e.g., 'question', 'item', 'modality', etc.)
    params : dict
        Parameters dictionary containing 'n' and 'sd' specifications
    coding_scheme : str, default="treatment"
        Coding scheme for categorical factors:
        - "treatment": dummy coding with first level as reference (Q-1 columns)
        - "sum": sum/deviation coding, sum to zero constraint (Q-1 columns)
        - "poly": polynomial/orthogonal coding for ordered factors (Q-1 columns)

    Returns
    -------
    np.ndarray
        Design matrix of shape (n_levels, n_columns) where:
        - If sd.{factor} is None: shape (n_levels, 0) - no slopes
        - If sd.{factor} is numeric: shape (n_levels, 1) - numeric coding
        - If sd.{factor} is list: shape (n_levels, len(list)) - categorical coding
    
    Notes
    -----
    - Numeric coding: mean-centered values [-2, -1, 0, +1, +2] for 5 levels
    - Categorical coding: depends on coding_scheme parameter
    - Missing sd specification returns empty matrix (no random slopes)
    """
    if factor_name not in params["n"]:
        raise ValueError(f"Factor '{factor_name}' not found in params['n']")
    
    n_levels = params["n"][factor_name]
    sdval = params["sd"].get(factor_name, None)
    
    if sdval is None:
        # no random slope specified for this factor
        return np.zeros((n_levels, 0), dtype=float)

    if isinstance(sdval, (float, int)):
        # Numeric coding
        x = np.arange(n_levels, dtype=float)
        x -= x.mean()  # mean-center
        return x.reshape(n_levels, 1)

    if isinstance(sdval, list):
        n_cols = len(sdval)
        
        # Validate list length for categorical coding
        if n_cols != (n_levels - 1) and coding_scheme in ["treatment", "sum", "poly"]:
            raise ValueError(
                f"For factor-coded '{factor_name}' with n.{factor_name}={n_levels} "
                f"and coding_scheme='{coding_scheme}', sd.{factor_name} must be "
                f"length {n_levels-1}, but got {n_cols}."
            )
        
        # Handle different coding schemes
        if coding_scheme == "treatment":
            # Treatment/dummy coding (reference = first level)
            codes = np.zeros((n_levels, n_levels - 1), dtype=float)
            for lev in range(1, n_levels):
                codes[lev, lev - 1] = 1.0
            return codes
            
        elif coding_scheme == "sum":
            # Sum/deviation coding (sum to zero)
            codes = np.zeros((n_levels, n_levels - 1), dtype=float)
            for lev in range(1, n_levels):
                codes[lev, lev - 1] = 1.0
            # Last level gets -1 for all columns
            codes[0, :] = -1.0
            return codes
            
        elif coding_scheme == "poly":
            # Polynomial/orthogonal coding for ordered factors
            codes = np.zeros((n_levels, n_levels - 1), dtype=float)
            x = np.arange(n_levels)
            
            for p in range(1, n_levels):
                # Generate polynomial contrast of degree p
                poly_vals = np.power(x - x.mean(), p)
                # Orthogonalize against previous polynomials if needed
                poly_vals = poly_vals - poly_vals.mean()
                # Normalize
                if np.std(poly_vals) > 0:
                    poly_vals = poly_vals / np.std(poly_vals)
                codes[:, p - 1] = poly_vals
            return codes
            
        else:
            # Custom coding: assume user provided explicit contrast matrix
            if n_cols == n_levels:
                # Full contrast matrix provided
                return np.array(sdval, dtype=float).reshape(n_levels, n_cols)
            else:
                # Assume treatment coding if unsure
                codes = np.zeros((n_levels, n_cols), dtype=float)
                for lev in range(min(n_levels - 1, n_cols)):
                    if lev + 1 < n_levels:
                        codes[lev + 1, lev] = 1.0
                return codes

    raise TypeError(f"sd.{factor_name} must be float/int or a list for factor coding.")

def get_factor_codes_dict(params: dict, factors: list = None) -> dict:
    """
    Get design matrices for multiple factors
    
    Parameters
    ----------
    params : dict
        Parameters dictionary
    factors : list, optional
        List of factor names. If None, uses all factors with sd specifications.
        
    Returns
    -------
    dict
        Dictionary mapping factor names to their design matrices
    """
    if factors is None:
        # Auto-detect factors from sd specifications
        factors = [f for f in params["sd"].keys() 
                  if f in params["n"] and params["sd"][f] is not None]
        # Remove non-factor entries
        factors = [f for f in factors if f not in ["error", "re_formula"]]

    return {factor: get_factor_codes(factor, params) for factor in factors}

def make_random_effects(params: dict):
    re_formula = params["sd"].get("re_formula")
    if not re_formula:
        raise ValueError("Missing 'sd.re_formula' in params['sd'].")

    formula = Formula(re_formula) if isinstance(re_formula, str) else re_formula
    random_effects = {}

    for term in formula:
        content = term.strip("()")
        lhs, group = content.split("|")
        lhs = lhs.strip()   
        group = group.strip() 

        n_key = f"{group}"
        if n_key not in params["n"]:
            raise ValueError(f"Missing 'n.{group}' for grouping factor '{group}'.")

        predictors = [t.strip() for t in lhs.split("+")]  
        has_intercept = ("1" in predictors)
        slopes = [p for p in predictors if p != "1"]

        # random intercept SD for that group
        intercept_sd_val = params["sd"].get(group, 0)
        intercept_sd = [float(intercept_sd_val)] if has_intercept else []

        # validate and gather slope structure
        slope_structure = _validate_random_effects_structure(slopes, params, group)
        slopes_sd = slope_structure['sd_values']

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

class Data(NamedTuple):
    """Data container for generated data
    
    Attributes
    ----------
    word: npt.ArrayLike
        Generated word data
    image: npt.ArrayLike
        Generated image data
    """
    word: np.ndarray
    image: np.ndarray

def generate(params: dict, seed: int = None):
    """
    Generate data using Generalized Linear Mixed Effects Model (GLMM)
    
    Returns
    -------
    np.ndarray: A (subjects, questions, items, modalities) matrix containing simulated values.
    """
    if seed is not None:
        np.random.seed(seed)

    # get family and link specifications
    family_name = params.get('glm').get('family', 'gaussian')
    link_name = params.get('glm').get('link', 'identity') 
    
    # validate and get family object
    validate_family_link_combination(family_name, link_name)
    family = get_family(family_name, link_name)
    
    # get distribution parameters
    family_params = params.get('glm').get('family_params', {})
    if family_params is None:
        family_params = {}
    else:
        family_params = family_params.copy()  # Don't modify original
    
    # calculate baseline linear predictor
    linear_predictor = calculate_baseline(params)

    # Add random effects
    random_effects = make_random_effects(params)
    n_subj, n_q, n_item, n_mod = linear_predictor.shape
    total_contrib = np.zeros_like(linear_predictor)
    
    # get factor coding matrices
    question_codes = get_factor_codes("question", params)
    m_question = question_codes.shape[1]
    question_codes_5d = question_codes.reshape(1, n_q, 1, 1, m_question)
    modality_code = np.array([0, 1]).reshape(1, 1, 1, n_mod)

    # parse formula and apply random effects
    formula = Formula(params["sd"]["re_formula"]) if isinstance(params["sd"]["re_formula"], str) else params["sd"]["re_formula"] 

    for term in formula:
        content = term.strip("()")
        lhs, group = content.split("|")
        lhs = lhs.strip()
        group = group.strip()

        # get the random effect matrix for this group
        re_matrix = random_effects.get(group, None)
        if re_matrix is None:
            continue

        effect_names = lhs.split("+")
        effect_names = [e.strip() for e in effect_names]
        factor_contrib = np.zeros_like(linear_predictor)

        # track which column of re_matrix we're using
        next_col = 0

        for p in effect_names:
            if p == "1":
                # Random intercept contribution
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
                    factor_contrib += slope_4d * modality_code
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
                # Handle any factor using its design matrix
                factor_codes = get_factor_codes(p, params)
                n_factor_cols = factor_codes.shape[1]
                
                if n_factor_cols == 0:
                    continue
                    
                # Apply slopes for each contrast column
                for col_idx in range(n_factor_cols):
                    if next_col >= re_matrix.shape[1]:
                        break
                        
                    slope_vals = re_matrix[:, next_col]
                    
                    # Create factor codes for broadcasting
                    if p == "question":
                        codes_broadcast = factor_codes[:, col_idx].reshape(1, n_q, 1, 1)
                    elif p == "item":
                        codes_broadcast = factor_codes[:, col_idx].reshape(1, 1, n_item, 1) 
                    elif p == "modality":
                        codes_broadcast = factor_codes[:, col_idx].reshape(1, 1, 1, n_mod)
                    else:
                        # Skip unknown factors
                        print(f"Warning: Unknown factor '{p}' - skipping slope application")
                        next_col += 1
                        continue
                    
                    # Apply slope based on grouping factor
                    if group == "subject":
                        slope_4d = slope_vals.reshape(n_subj, 1, 1, 1)
                    elif group == "question":
                        slope_4d = slope_vals.reshape(1, n_q, 1, 1)
                    elif group == "item":
                        slope_4d = slope_vals.reshape(1, 1, n_item, 1)
                    elif group == "modality":
                        slope_4d = slope_vals.reshape(1, 1, 1, n_mod)
                    else:
                        next_col += 1
                        continue
                        
                    factor_contrib += slope_4d * codes_broadcast
                    next_col += 1

        total_contrib += factor_contrib

    # Complete linear predictor
    eta = linear_predictor + total_contrib
    
    # Add residual error to gaussian
    if family_name == 'gaussian' and params["sd"].get("error") is not None:
        family_params['sigma'] = params["sd"]["error"]
    
    # Transform to mean scale
    mu = family.link.inverse_link(eta)
    
    # numerical stability bounds for non-Gaussian families with identity link
    if family_name != 'gaussian' and link_name == 'identity':
        
        min_rt = 50.0    
        max_rt = 10000.0
        
        n_extreme = np.sum((mu < min_rt) | (mu > max_rt))
        if n_extreme > 0:
            n_total = mu.size
            pct_extreme = 100 * n_extreme / n_total
            
            warnings.warn(
                f"Simulation stability: {n_extreme}/{n_total} ({pct_extreme:.1f}%) "
                f"μ values are outside stable range [{min_rt}, {max_rt}] ms. "
                f"Range: [{np.min(mu):.1f}, {np.max(mu):.1f}] ms. "
                f"Applying bounds to prevent R convergence issues. "
                f"Consider: (1) reducing random effect SDs, (2) increasing baseline RTs, "
                f"or (3) using log link for better numerical properties.",
                UserWarning
            )
        
        mu = np.clip(mu, min_rt, max_rt)
    
    
    # Extract shift parameters for RT modeling
    shift = params.get('sd').get('shift', None)
    shift_noise = params.get('sd').get('shift_noise', None)

    # Generate observations from the specified family: Y ~ Family(μ, θ)
    simulated_data = family.simulate(mu, shift=shift, shift_noise=shift_noise, **family_params)
    
    return simulated_data

class DataGenerator(object):
    """Data generator
    
    Methods
    -------
    fit_transform(self, params:dict=None, overwrite:bool=False, seed:int=None, verbose:bool=False)
        Generate data based on parameters

    to_pandas(self) -> pd.DataFrame
        Convert data to pandas dataframe

    Attributes
    ----------
    data: Data
        Generated data container with word and image arrays
        
    word: np.ndarray
        Generated word data (convenience accessor for data.word)

    image: np.ndarray
        Generated image data (convenience accessor for data.image)
        
    params: dict
        Current parameter configuration
    """ 

    def __init__(self):
        if config.p is not None:
            self.params = copy.deepcopy(config.p)
        else:
            self.params = {}
            
        # check GLMM parameters are initialized
        if 'family' not in self.params:
            glmm_defaults = get_glmm_defaults()
            self.params.update(glmm_defaults)
            
        self.data = None
        self.word = None
        self.image = None

    def fit_transform(self, params: dict = None, overwrite: bool = False, seed: int = None, verbose: bool = False):
        """
        Generate data based on the current (or new) params. 
        If overwrite=True, we replace self.params with new ones.
        If not, we only partially update them or keep them as is.
        """
        if verbose:
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
                raw_data = generate(self.params, seed=seed)
                self.data = Data(word=raw_data[:, :, :, 0], image=raw_data[:, :, :, 1])
                self.word = self.data.word
                self.image = self.data.image
            elif params is not None and len(params) == len(self.params):
                validate_params(params)
                self.params = parse_params(params)
                raw_data = generate(self.params, seed=seed)
                self.data = Data(word=raw_data[:, :, :, 0], image=raw_data[:, :, :, 1])
                self.word = self.data.word
                self.image = self.data.image
        else:
            if params is not None and len(params) == len(self.params):
                validate_params(params)
                gen_params = parse_params(params)
                raw_data = generate(gen_params, seed=seed)
                self.data = Data(word=raw_data[:, :, :, 0], image=raw_data[:, :, :, 1])
                self.word = self.data.word
                self.image = self.data.image
            elif params is not None and len(params) != len(self.params):
                # partial update
                updated = update_params(self.params, params)
                raw_data = generate(updated, seed=seed)
                self.data = Data(word=raw_data[:, :, :, 0], image=raw_data[:, :, :, 1])
                self.word = self.data.word
                self.image = self.data.image
            else:
                # use the existing self.params
                raw_data = generate(self.params, seed=seed)
                self.data = Data(word=raw_data[:, :, :, 0], image=raw_data[:, :, :, 1])
                self.word = self.data.word
                self.image = self.data.image
        return self

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert data => long-format DataFrame with columns:
        [subject, rt, question, item, modality].
        """
        if self.data is None:
            raise ValueError("No data generated yet. Call fit_transform first.")

        # word data
        n_participants, n_questions, n_items = self.data.word.shape
        word_df = pd.DataFrame({
            "subject": np.repeat(np.arange(n_participants), n_questions * n_items),
            "rt": self.data.word.flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_items), n_participants),
            "item": np.tile(np.arange(n_items), n_participants * n_questions),
            "modality": "word"
        })

        # image data
        n_participants, n_questions, n_items = self.data.image.shape
        image_df = pd.DataFrame({
            "subject": np.repeat(np.arange(n_participants), n_questions * n_items),
            "rt": self.data.image.flatten(),
            "question": np.tile(np.repeat(np.arange(n_questions), n_items), n_participants),
            "item": np.tile(np.arange(n_items), n_participants * n_questions),
            "modality": "image"
        })

        df = pd.concat([image_df, word_df], ignore_index=True)
        return df

    def get_data(self) -> Data:
        """
        Get the Data container with word and image arrays.
        
        Returns
        -------
        Data
            NamedTuple containing word and image arrays
            
        Raises
        ------
        ValueError
            If no data has been generated yet
        """
        if self.data is None:
            raise ValueError("No data generated yet. Call fit_transform first.")
        return self.data
    
    def get_word_data(self) -> np.ndarray:
        """
        Get word modality data.
        
        Returns
        -------
        np.ndarray
            Word data with shape (subjects, questions, items)
        """
        if self.data is None:
            raise ValueError("No data generated yet. Call fit_transform first.")
        return self.data.word
    
    def get_image_data(self) -> np.ndarray:
        """
        Get image modality data.
        
        Returns
        -------
        np.ndarray
            Image data with shape (subjects, questions, items)
        """
        if self.data is None:
            raise ValueError("No data generated yet. Call fit_transform first.")
        return self.data.image

    def summary(self) -> dict:
        """
        Get summary statistics of the generated data.
        
        Returns
        -------
        dict
            Summary statistics for word and image data, plus model info
        """
        if self.data is None:
            raise ValueError("No data generated yet. Call fit_transform first.")
        
        # Get family information
        family_info = self.get_family_info()
        
        # Get effective family parameters (including defaults)
        effective_family_params = family_info['current']['family_params'].copy()
        family_name = family_info['current']['family']
        
        # For inverse Gaussian, show default lambda if not specified
        if family_name == 'inverse_gaussian':
            # Check both possible parameter names
            has_lambda = ('lambda' in effective_family_params and effective_family_params.get('lambda') is not None)
            has_lambda_param = ('lambda_param' in effective_family_params and effective_family_params.get('lambda_param') is not None)
            
            if not has_lambda and not has_lambda_param:
                # Calculate representative default lambda using mean RT
                mean_word_rt = float(np.mean(self.data.word))
                mean_image_rt = float(np.mean(self.data.image))
                mean_rt = (mean_word_rt + mean_image_rt) / 2
                
                # Import utils to use lsolve
                from . import utils
                default_lambda = float(utils.lsolve(mean_rt))
                
                effective_family_params['lambda'] = f"auto (≈{default_lambda:.1f} based on mean RT={mean_rt:.1f})"
                effective_family_params['_lambda_explanation'] = "Uses utils.lsolve(mu) = mu^2 for canonical parameterization"
            
        return {
            'model': {
                'family': family_info['current']['family'],
                'link': family_info['current']['link'],
                'family_params_specified': family_info['current']['family_params'],
                'family_params_effective': effective_family_params,
                'description': family_info['description']
            },
            'data': {
                'word': {
                    'shape': self.data.word.shape,
                    'mean': float(np.mean(self.data.word)),
                    'std': float(np.std(self.data.word)),
                    'min': float(np.min(self.data.word)),
                    'max': float(np.max(self.data.word)),
                    'median': float(np.median(self.data.word)),
                    'skewness': float(stats.skew(self.data.word.flatten()))
                },
                'image': {
                    'shape': self.data.image.shape,
                    'mean': float(np.mean(self.data.image)),
                    'std': float(np.std(self.data.image)),
                    'min': float(np.min(self.data.image)),
                    'max': float(np.max(self.data.image)),
                    'median': float(np.median(self.data.image)),
                    'skewness': float(stats.skew(self.data.image.flatten()))
                }
            }
        }

    def set_family(self, family: str, link: str, family_params: dict = None) -> 'DataGenerator':
        """
        Set the distribution family and link function for GLMM simulation.
        
        Parameters
        ----------
        family : str
            Distribution family: 'gaussian', 'gamma', 'inverse_gaussian', 'lognormal'
        link : str  
            Link function: 'identity', 'log', 'inverse', 'sqrt'
        family_params : dict, optional
            Parameters for the distribution family
            
        Returns
        -------
        DataGenerator
            Self for method chaining
        """
        # Validate combination
        validate_family_link_combination(family, link)
        
        # Set parameters
        self.params['family'] = family
        self.params['link'] = link
        
        if family_params is None:
            family_params = get_default_family_params(family)
        
        self.params['family_params'] = family_params
        
        return self
    
    def get_family_info(self) -> dict:
        """
        Get information about the current family and link settings.
        
        Returns
        -------
        dict
            Information about family, link, and parameters
        """
        family_name = self.params.get('glm').get('family', 'gaussian')
        link_name = self.params.get('glm').get('link', 'identity')
        family_params = self.params.get('glm').get('family_params', {})
        
        # Get recommended params for comparison
        recommended = get_recommended_family_params_for_rt()
        
        return {
            'current': {
                'family': family_name,
                'link': link_name,
                'family_params': family_params
            },
            'recommended_for_rt': recommended,
            'description': f"Using {family_name} family with {link_name} link"
        }
    
    def set_rt_family(self, rt_type: str = 'gamma') -> 'DataGenerator':
        """
        Convenience method to set up recommended family/link for RT data.
        
        Parameters
        ----------
        rt_type : str
            Type of RT distribution: 'gamma', 'inverse_gaussian', 'lognormal', 'gaussian'
            
        Returns
        -------
        DataGenerator
            Self for method chaining
        """
        # Import here to avoid circular imports
        from .params import get_rt_glmm_config
        
        try:
            config = get_rt_glmm_config(rt_type)
            return self.set_family(config['family'], config['link'], config['family_params'])
        except ValueError as e:
            raise ValueError(f"Error setting RT family: {e}")

def get_default_family_params(family_name: str) -> dict:
    """
    Get default parameters for distribution families.
    
    Parameters
    ----------
    family_name : str
        Name of the distribution family
        
    Returns
    -------
    dict
        Default parameters for the family
    """
    return DEFAULT_FAMILY_PARAMS.get(family_name, {}).copy()

def get_recommended_family_params_for_rt() -> dict:
    """
    Get recommended family parameters for reaction time data.
    
    Returns
    -------
    dict
        Recommended parameters for different RT distributions
    """
    return {
        family_name: {
            **{k: v for k, v in config.items() if k != 'description'},
            'description': config.get('description', f'Recommended configuration for {family_name}')
        }
        for family_name, config in RT_FAMILY_CONFIGS.items()
    }

def _validate_random_effects_structure(slopes: list, params: dict, group: str) -> dict:
    """
    Validate and organize random effects structure for a grouping factor.
    
    Parameters
    ----------
    slopes : list
        List of slope factor names (e.g., ['question', 'modality'])
    params : dict
        Parameters dictionary
    group : str
        Grouping factor name (e.g., 'subject', 'item')
        
    Returns
    -------
    dict
        Information about the random effects structure:
        - 'slope_info': list of dicts with factor info
        - 'total_slopes': total number of slope parameters
        - 'sd_values': flattened list of SD values
    """
    slope_info = []
    total_slopes = 0
    sd_values = []
    
    for factor in slopes:
        # Get design matrix for this factor
        try:
            factor_codes = get_factor_codes(factor, params)
            n_cols = factor_codes.shape[1]
        except (ValueError, KeyError):
            # Factor not found or no slopes specified
            n_cols = 0
            
        # Get SD specification
        sd_val = params["sd"].get(factor, 0)
        
        if isinstance(sd_val, list):
            n_sd_params = len(sd_val)
            factor_sd_values = [float(x) for x in sd_val]
        else:
            n_sd_params = 1 if sd_val != 0 else 0
            factor_sd_values = [float(sd_val)] if sd_val != 0 else []
            
        # Validate consistency
        if n_cols > 0 and n_sd_params != n_cols:
            raise ValueError(
                f"Group '{group}', factor '{factor}': "
                f"Design matrix has {n_cols} columns but "
                f"sd.{factor} specifies {n_sd_params} parameters. "
                f"For categorical factors, provide a list of {n_cols} SD values."
            )
            
        slope_info.append({
            'factor': factor,
            'n_cols': n_cols,
            'n_sd_params': n_sd_params,
            'sd_values': factor_sd_values,
            'is_categorical': isinstance(sd_val, list)
        })
        
        total_slopes += n_cols
        sd_values.extend(factor_sd_values)
    
    return {
        'slope_info': slope_info,
        'total_slopes': total_slopes,
        'sd_values': sd_values
    }
