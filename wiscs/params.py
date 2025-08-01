from collections import defaultdict
import numpy as np
import numpy.typing as npt
from typing import Callable, Union, get_args
from .formula import Formula

EMPTY_PARAMS = {
    'word.perceptual': Union[int, float],
    'image.perceptual': Union[int, float],

    'word.conceptual': Union[int, float],
    'image.conceptual': Union[int, float],

    'word.task': Union[npt.ArrayLike, Callable[..., npt.ArrayLike]],
    'image.task': Union[npt.ArrayLike, Callable[..., npt.ArrayLike]],

    'sd.item': Union[npt.ArrayLike, None, int, float, dict, list],
    'sd.question': Union[npt.ArrayLike, None, int, float, dict, list],
    'sd.subject': Union[npt.ArrayLike, None, int, float, dict, list],
    'sd.modality': Union[npt.ArrayLike, None, int, float, dict, list],
    'sd.re_formula':Union[str, Formula],
    "sd.error": Union[int, float, None],
    "sd.shift": Union[int, float, None],  # Minimum RT shift (adds to all generated RTs)
    "sd.shift_noise": Union[int, float, None],  # Subject-level Gaussian noise

    "corr.subject":Union[npt.ArrayLike, None, int, float, dict],
    "corr.question":Union[npt.ArrayLike, None, int, float, dict],
    "corr.item":Union[npt.ArrayLike, None, int, float, dict],
    "corr.modality":Union[npt.ArrayLike, None, int, float, dict],

    'n.subject': int,
    'n.question': int,
    'n.item': int,
    
    # GLMM parameters
    'glm.family': str,  # Distribution family: 'gaussian', 'gamma', 'inverse_gaussian', 'lognormal'
    'glm.link': str,    # Link function: 'identity', 'log', 'inverse', 'sqrt'
    'glm.family_params': Union[dict, None],  # Family-specific parameters

}

# Valid GLMM combinations
VALID_FAMILIES = ['gaussian', 'gamma', 'inverse_gaussian', 'lognormal']
VALID_LINKS = ['identity', 'log', 'inverse', 'sqrt']

# Valid family-link combinations
VALID_FAMILY_LINK_COMBINATIONS = {
    'gaussian': ['identity', 'log'],
    'gamma': ['log', 'inverse', 'identity'],
    'inverse_gaussian': ['inverse', 'log', 'identity'],
    'lognormal': ['log', 'identity']
}

# Default family parameters
DEFAULT_FAMILY_PARAMS = {
    'gaussian': {'sigma': 1.0},
    'gamma': {'shape': 2.0},
    'inverse_gaussian': {'lambda': 1.0},
    'lognormal': {'sigma': 0.25}
}

# RT-optimized family parameter configurations
RT_FAMILY_CONFIGS = {
    'gamma': {
        'shape': 4.0,  # Shape parameter (higher = less skewed, more normal-like)
        'description': 'Gamma distribution with log link - good for right-skewed RT data'
    },
    'inverse_gaussian': {
        'lambda': 2.0,  # Dispersion parameter (higher = less dispersed)
        'description': 'Inverse Gaussian with inverse link - theoretical RT distribution'
    },
    'lognormal': {
        'sigma': 0.3,  # Log-scale standard deviation (smaller = less skewed)
        'description': 'Log-normal distribution with log link - simple RT model'
    },
    'gaussian': {
        'sigma': 100.0,  # Standard deviation in RT units (ms)
        'description': 'Gaussian distribution with identity link - traditional LMM approach'
    }
}

# Default shift parameters for realistic RT distributions
DEFAULT_SHIFT_PARAMS = {
    'shift': 200.0,       # 200ms minimum RT (typical lower bound for human responses)
    'shift_noise': 50.0   # 50ms subject-level variation in minimum RT
}

def validate_params(params: dict) -> bool:
    """
    Validate a dictionary of parameters against the expected structure and types in EMPTY_PARAMS.

    Parameters
    ----------
    params: dict
        The parameters to validate.

    Returns
    -------
    bool
        True if all parameters are valid; raises an exception otherwise.

    Raises
    ------
    ValueError
        If a parameter is unexpected or does not match the expected type.
    TypeError
        If a parameter value does not match the expected type.
    """
    for key, value in params.items():
        # Match exact keys or wildcard patterns
        if key not in EMPTY_PARAMS:
            matched = False
            for prefix, expected_type in EMPTY_PARAMS.items():
                if "*" in prefix and key.startswith(prefix.split('.')[0]):
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unexpected parameter: {key}")
        else:
            expected_type = EMPTY_PARAMS[key]

        # Check the type of the value
        if isinstance(expected_type, type):
            # Simple type
            if not isinstance(value, expected_type):
                raise TypeError(f"Parameter {key} should be {expected_type}, but got {type(value)}")
        elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            # Union types - check each possibility
            union_args = get_args(expected_type)
            valid = False
            
            for arg in union_args:
                # Check for sequence types (lists, tuples) - look for _NestedSequence
                if '_NestedSequence' in str(arg):
                    if isinstance(value, (list, tuple, np.ndarray)):
                        valid = True
                        break
                # Check for Callable
                elif hasattr(arg, '__origin__') and str(arg.__origin__).startswith('typing.Callable'):
                    if callable(value):
                        valid = True
                        break
                # Check for simple types
                elif isinstance(arg, type):
                    if isinstance(value, arg):
                        valid = True
                        break
                # Check for None type
                elif arg is type(None) and value is None:
                    valid = True
                    break
            
            if not valid:
                # Try a more permissive check for common array-like types
                if isinstance(value, (list, tuple, np.ndarray)) and any('_NestedSequence' in str(arg) for arg in union_args):
                    valid = True
                elif callable(value) and any('Callable' in str(arg) for arg in union_args):
                    valid = True
                    
            if not valid:
                raise TypeError(f"Parameter {key} should be array-like (list, tuple, numpy array) or callable, but got {type(value)}")
        elif str(expected_type).startswith('numpy._typing') and 'ArrayLike' in str(expected_type):
            # Array-like types
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise TypeError(f"Parameter {key} should be array-like, but got {type(value)}")
        elif expected_type == Callable[..., npt.ArrayLike]:
            # Callable returning ArrayLike
            if not callable(value):
                raise TypeError(f"Parameter {key} should be callable, but got {type(value)}")
            # Optionally: validate the callable's return type if it can be executed with dummy input
            try:
                dummy_output = value()
                if not isinstance(dummy_output, (list, tuple, np.ndarray)):
                    raise TypeError(f"Callable for {key} should return array-like, but returned {type(dummy_output)}")
            except Exception as e:
                raise ValueError(f"Callable for {key} raised an exception during validation: {e}")
        else:
            raise TypeError(f"Unsupported type for parameter {key}: {expected_type}")
        
        # check that n_questions is equal to length of word.task and image.task
        if key == 'n.question':
            if len(params['word.task']) != value or len(params['image.task']) != value:
                raise ValueError(f"n.question should be equal to length of word.task and image.task")

    # GLMM-specific validation
    _validate_glmm_params(params)
    
    # Validate shift parameters
    _validate_shift_params(params)
    
    return True

def _validate_glmm_params(params: dict) -> None:
    """
    Validate GLMM-specific parameters.
    
    Parameters
    ----------
    params : dict
        Parameters dictionary
        
    Raises
    ------
    ValueError
        If GLMM parameters are invalid
    """
    # Get GLMM parameters with defaults
    family = params.get('family', 'gaussian')
    link = params.get('link', 'identity')
    family_params = params.get('family_params', None)
    
    # Validate family
    if family not in VALID_FAMILIES:
        raise ValueError(
            f"Invalid family '{family}'. Valid families: {VALID_FAMILIES}"
        )
    
    # Validate link
    if link not in VALID_LINKS:
        raise ValueError(
            f"Invalid link '{link}'. Valid links: {VALID_LINKS}"
        )
    
    # Validate family-link combination
    if link not in VALID_FAMILY_LINK_COMBINATIONS[family]:
        raise ValueError(
            f"Invalid link '{link}' for family '{family}'. "
            f"Valid links for {family}: {VALID_FAMILY_LINK_COMBINATIONS[family]}"
        )
    
    # Validate family parameters
    if family_params is not None:
        _validate_family_params(family, family_params)


def _validate_family_params(family: str, family_params: dict) -> None:
    """
    Validate family-specific parameters.
    
    Parameters
    ----------
    family : str
        Distribution family name
    family_params : dict
        Family-specific parameters
        
    Raises
    ------
    ValueError
        If family parameters are invalid
    """
    if family == 'gaussian':
        if 'sigma' in family_params:
            sigma = family_params['sigma']
            if not isinstance(sigma, (int, float)) or sigma <= 0:
                raise ValueError("Gaussian family: 'sigma' must be a positive number")
                
    elif family == 'gamma':
        if 'shape' in family_params:
            shape = family_params['shape']
            if not isinstance(shape, (int, float)) or shape <= 0:
                raise ValueError("Gamma family: 'shape' must be a positive number")
                
    elif family == 'inverse_gaussian':
        if 'lambda' in family_params:
            lambda_param = family_params['lambda']
            # Allow None for automatic lambda calculation using utils.lsolve(mu)
            if lambda_param is not None:
                if not isinstance(lambda_param, (int, float)) or lambda_param <= 0:
                    raise ValueError("Inverse Gaussian family: 'lambda' must be a positive number or None (for automatic calculation)")
                
    elif family == 'lognormal':
        if 'sigma' in family_params:
            sigma = family_params['sigma']
            if not isinstance(sigma, (int, float)) or sigma <= 0:
                raise ValueError("Log-Normal family: 'sigma' must be a positive number")

def parse_params(params):
    """
    Parse a dictionary with compound keys into a nested dictionary.

    Parameters
    ----------
    params: dict 
        Dictionary with keys in the format 'category.attribute'.

    Returns
    -------
    dict: Nested dictionary with categories as top-level keys and attributes as subkeys.
    """
    parsed = defaultdict(dict)
    for key, value in params.items():
        try:
            category, attribute = key.split('.')
            parsed[category][attribute] = value
        except ValueError:
            parsed[key] = value
    return dict(parsed)

def update_params(params, kwargs) -> dict:
    """
    Update parameters with new values.

    Parameters
    ----------
    params: dict
        Original dictionary of parameters

    kwargs: dict
        Keys and values to be updated

    Returns
    -------
    dict: Updated parameters.
    """
    update = params.copy()
    new = parse_params(kwargs)

    for key, subdict in new.items():
        if key in update and isinstance(update[key], dict) and isinstance(subdict, dict):
            update[key].update(subdict)
        else:
            update[key] = subdict

    return update

def _validate_shift_params(params: dict) -> None:
    """
    Validate shift-related parameters.
    
    Parameters
    ----------
    params : dict
        Parameters dictionary
        
    Raises
    ------
    ValueError
        If shift parameters are invalid
    """
    shift = params.get('shift', None)
    shift_noise = params.get('shift_noise', None)
    
    # Validate shift parameter
    if shift is not None:
        if not isinstance(shift, (int, float)):
            raise ValueError("'shift' must be a number")
        if shift < 0:
            raise ValueError("'shift' must be non-negative")
    
    # Validate shift_noise parameter
    if shift_noise is not None:
        if not isinstance(shift_noise, (int, float)):
            raise ValueError("'shift_noise' must be a number")
        if shift_noise < 0:
            raise ValueError("'shift_noise' must be non-negative")