from collections import defaultdict
import numpy as np
import numpy.typing as npt
from typing import Callable, Union, get_args, Dict
import types

EMPTY_PARAMS = {
    'word.perceptual': Union[int, float],
    'image.perceptual': Union[int, float],

    'word.concept': Union[int, float],
    'image.concept': Union[int, float],

    'word.task': Union[npt.ArrayLike, Callable[..., npt.ArrayLike]],
    'image.task': Union[npt.ArrayLike, Callable[..., npt.ArrayLike]],

    'image.*': Union[int, float, Callable[..., npt.ArrayLike]],
    'word.*': Union[int, float, Callable[..., npt.ArrayLike]],

    'var.image': Union[int, float],
    'var.word': Union[int, float],
    'var.question': Union[int, float],
    'var.participant': Union[int, float], 

    'n.participant': int,
    'n.question': int,
    'n.trial': int,

    'design': Dict[str, Union[str, bool]],
}

def validate_params(params: dict) -> bool:
    for key, value in params.items():
        if key not in EMPTY_PARAMS:
            raise ValueError(f"Unexpected parameter: {key}")

        expected_type = EMPTY_PARAMS[key]

        # Handle Union types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            union_args = get_args(expected_type)
            # Special case for npt.ArrayLike in Union
            if npt.ArrayLike in union_args and isinstance(value, np.ndarray):
                continue
            if not any(isinstance(value, t) for t in union_args if isinstance(t, type)):
                raise TypeError(f"Parameter {key} should be one of {union_args}, but got {type(value)}")

        # handle npt.ArrayLike
        elif expected_type == npt.ArrayLike:
            if not isinstance(value, (np.ndarray, list, tuple)):
                raise TypeError(f"Parameter {key} should be array-like, but got {type(value)}")

        # Handle other types (e.g., dict, Callable, etc.)
        elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is dict:
            key_type, value_type = get_args(expected_type)
            if not isinstance(value, dict):
                raise TypeError(f"Parameter {key} should be a dict, but got {type(value)}")
            for k, v in value.items():
                if not isinstance(k, key_type):
                    raise TypeError(f"Keys in parameter {key} should be {key_type}, but got {type(k)}")
                if not isinstance(v, value_type):
                    raise TypeError(f"Values in parameter {key} should be {value_type}, but got {type(v)}")

        # Handle Callable types
        elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is Callable:
            if not callable(value):
                raise TypeError(f"Parameter {key} should be callable, but got {type(value)}")

        else:
            raise TypeError(f"Unsupported type for parameter {key}: {expected_type}")

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