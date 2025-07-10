from .params import EMPTY_PARAMS
from .params import *
from .simulate import *
from . import config
from .utils import *
from .formula import Formula
from .plotting import * 
from .glm import (
    LinkFunction, DistributionFamily,
    IdentityLink, LogLink, InverseLink, SqrtLink,
    GaussianFamily, GammaFamily, InverseGaussianFamily, LogNormalFamily,
    get_link_function, get_family, validate_family_link_combination,
    get_available_families, get_available_links, create_family_summary
)

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

def set_re_method(method:str):
    """Set the method for generating random effects"""
    if method in ["cholesky", "eigen"]:
        config.re_method = method
    else:
        raise ValueError("Method must be 'cholesky' or 'eigen'")