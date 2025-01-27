from collections import defaultdict
import numpy as np
import numpy.typing as npt
from typing import Callable, Union, get_args

EMPTY_PARAMS = {
    'word.perceptual': Union[int, float],
    'image.perceptual': Union[int, float],

    'word.conceptual': Union[int, float],
    'image.conceptual': Union[int, float],

    'word.task': Union[npt.ArrayLike, Callable[..., npt.ArrayLike]],
    'image.task': Union[npt.ArrayLike, Callable[..., npt.ArrayLike]],

    'sd.item': Union[npt.ArrayLike,None],
    'sd.question':Union[npt.ArrayLike,None],
    'sd.subject': Union[npt.ArrayLike, None, int, float],
    "sd.error": Union[int, float],

    'n.subject': int,
    'n.question': int,
    'n.item': int,
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
            # Union types
            union_args = get_args(expected_type)
            if not any(isinstance(value, t) for t in union_args if isinstance(t, type)):
                raise TypeError(f"Parameter {key} should be one of {union_args}, but got {type(value)}")
        elif expected_type == npt.ArrayLike:
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

    return True

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