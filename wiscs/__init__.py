from .params import EMPTY_PARAMS
from .params import *
from .simulate import *
from . import config
from .utils import *
from .formula import *
from .plotting import * 

import sys

def set_params(params:dict=None, return_empty=False, verbose:bool=True):
    """Set data parameters"""

    if params is None and return_empty:
        sys.stdout.write(
            f"Params must be a dictionary with the following keys:\n {EMPTY_PARAMS.keys()}"
        )
        return EMPTY_PARAMS
    elif params is not None and return_empty:
        raise ValueError("If params is provided, return_empty must be False")
    elif validate_params(params):
        config.p = parse_params(params)
        if verbose:
            print("Params set successfully")