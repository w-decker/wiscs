"""
Generalized Linear Model (GLM) Components for WISCS

This module provides the link functions and distribution families for 
Generalized Linear Mixed Effects Models (GLMM) in reaction time simulation.

Classes
-------
LinkFunction : ABC
    Abstract base class for link functions
DistributionFamily : ABC  
    Abstract base class for distribution families

Link Functions
--------------
IdentityLink
LogLink
InverseLink
SqrtLink    

Distribution Families
--------------------
GaussianFamily : Normal/Gaussian distribution
GammaFamily : Gamma distribution 
InverseGaussianFamily : Inverse Gaussian distribution 
LogNormalFamily : Log-Normal distribution 

Factory Functions
-----------------
get_link_function : Get link function by name
get_family : Get distribution family with specified link
validate_family_link_combination : Validate family/link combinations
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any
import warnings

from .params import VALID_FAMILY_LINK_COMBINATIONS
from . import utils

# Base Classes
class LinkFunction(ABC):
    """Abstract base class for link functions
    
    Link functions transform between the mean scale (μ) and linear predictor scale (η).
    The relationship is: g(μ) = η, where g is the link function.
    """
    
    @staticmethod
    @abstractmethod
    def link(mu: np.ndarray) -> np.ndarray:
        """Transform from mean scale to linear predictor scale

        Parameters
        ----------
        mu : np.ndarray
            Values on the mean scale (response scale)
            
        Returns
        -------
        np.ndarray
            Values on the linear predictor scale
        """
        pass
    
    @staticmethod
    @abstractmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        """Transform from linear predictor scale to mean scale

        Parameters
        ----------
        eta : np.ndarray
            Values on the linear predictor scale
            
        Returns
        -------
        np.ndarray
            Values on the mean scale (response scale)
        """
        pass
    
    @staticmethod
    @abstractmethod
    def derivative(mu: np.ndarray) -> np.ndarray:
        """Derivative of link function:
        
        Parameters
        ----------
        mu : np.ndarray
            Values on the mean scale
            
        Returns
        -------
        np.ndarray
            Derivative values
        """
        pass

class DistributionFamily(ABC):
    """Abstract base class for distribution families
    
    Distribution families define the probability distribution of the response variable
    in a GLM/GLMM, along with the associated variance function.
    """
    
    def __init__(self, link: LinkFunction):
        """Initialize with a link function
        
        Parameters
        ----------
        link : LinkFunction
            The link function to use with this family
        """
        self.link = link
    
    @abstractmethod
    def simulate(self, mu: np.ndarray, **kwargs) -> np.ndarray:
        """Generate random samples from the distribution
        
        Parameters
        ----------
        mu : np.ndarray
            Mean parameter values
        **kwargs
            Family-specific parameters
            
        Returns
        -------
        np.ndarray
            Random samples from the distribution
        """
        pass
    
    def mean(self, eta: np.ndarray) -> np.ndarray:
        """Expected value given linear predictor
        
        Parameters
        ----------
        eta : np.ndarray
            Linear predictor values
            
        Returns
        -------
        np.ndarray
            Expected values
        """
        return self.link.inverse_link(eta)
    
    @abstractmethod
    def variance(self, mu: np.ndarray, **kwargs) -> np.ndarray:
        """Variance function
        
        Parameters
        ----------
        mu : np.ndarray
            Mean parameter values
        **kwargs
            Family-specific parameters
            
        Returns
        -------
        np.ndarray
            Variance values
        """
        pass

# Link Function Implementations
class IdentityLink(LinkFunction):
    """Identity link
    """
    
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        return mu
    
    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        return eta
    
    @staticmethod
    def derivative(mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

class LogLink(LinkFunction):
    """Log link
    """
    
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(mu, 1e-10))  # Avoid log(0)
    
    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        return np.exp(np.minimum(eta, 700))  # Avoid overflow
    
    @staticmethod
    def derivative(mu: np.ndarray) -> np.ndarray:
        return 1.0 / np.maximum(mu, 1e-10)

class InverseLink(LinkFunction):
    """Inverse link
    """
    
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        return 1.0 / np.maximum(mu, 1e-10)
    
    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        return 1.0 / np.maximum(eta, 1e-10)
    
    @staticmethod
    def derivative(mu: np.ndarray) -> np.ndarray:
        return -1.0 / np.maximum(mu**2, 1e-20)


class SqrtLink(LinkFunction):
    """Square root link"""
    
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(mu, 0))
    
    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        return np.maximum(eta, 0)**2
    
    @staticmethod
    def derivative(mu: np.ndarray) -> np.ndarray:
        return 0.5 / np.sqrt(np.maximum(mu, 1e-10))

# Distribution Family Implementations
class GaussianFamily(DistributionFamily):
    """Normal/Gaussian distribution family

    Parameters
    ----------
    sigma : float, default=1.0
        Standard deviation parameter
    """
    
    def simulate(self, mu: np.ndarray, sigma: float = 1.0, shift: float = 0.0, 
                 shift_noise: float = 0.0, **kwargs) -> np.ndarray:
        """Generate samples from Gaussian distribution with optional shift
        
        Parameters
        ----------
        shift : float, default=0.0
            Minimum response time shift (e.g., 200ms for motor response time)
        shift_noise : float, default=0.0
            Standard deviation of random noise added to shift for each subject
        """
        samples = np.random.normal(mu, sigma, size=mu.shape)
        
        if shift > 0 or shift_noise > 0:
            # Apply shift with optional subject-level noise
            if shift_noise > 0:
                # Get unique subjects (assume first dimension is subjects)
                if samples.ndim >= 1:
                    n_subjects = samples.shape[0]
                    subject_shifts = shift + np.random.normal(0, shift_noise, n_subjects)
                    # Broadcast shift across other dimensions
                    shift_shape = [n_subjects] + [1] * (samples.ndim - 1)
                    applied_shift = subject_shifts.reshape(shift_shape)
                else:
                    applied_shift = shift + np.random.normal(0, shift_noise)
            else:
                applied_shift = shift
            
            samples += applied_shift
        
        return samples
    
    def variance(self, mu: np.ndarray, sigma: float = 1.0, **kwargs) -> np.ndarray:
        return np.full_like(mu, sigma**2)

class GammaFamily(DistributionFamily):
    """Gamma distribution family
    
    Parameters
    ----------
    shape : float, default=1.0
        Shape parameter. Higher values reduce skewness.
    """
    
    def simulate(self, mu: np.ndarray, shape: float = 1.0, shift: float = 0.0,
                 shift_noise: float = 0.0, **kwargs) -> np.ndarray:
        """Generate samples from Gamma distribution with optional shift
        
        Parameters
        ----------
        shift : float, default=0.0
            Minimum response time shift
        shift_noise : float, default=0.0
            Standard deviation of random noise added to shift for each subject
        """
        # Check for non-positive means
        if np.any(mu <= 0):
            n_negative = np.sum(mu <= 0)
            n_total = mu.size
            pct_negative = 100 * n_negative / n_total
            
            warnings.warn(
                f"GammaFamily.simulate: {n_negative}/{n_total} ({pct_negative:.1f}%) "
                f"mean values are non-positive (min={np.min(mu):.3f}). "
                f"This often occurs with identity link when random effects are large "
                f"relative to baseline. Consider: (1) increasing baseline values, "
                f"(2) reducing random effect SDs, or (3) using log link instead.",
                UserWarning
            )
            
            # Clip to small positive values to prevent crash
            mu = np.maximum(mu, 1e-6)
        
        # Gamma parameterization: shape=α, scale=μ/α
        scale = mu / shape
        samples = np.random.gamma(shape, scale, size=mu.shape)
        
        if shift > 0 or shift_noise > 0:
            # Apply shift with optional subject-level noise
            if shift_noise > 0:
                # Get unique subjects (assume first dimension is subjects)
                if samples.ndim >= 1:
                    n_subjects = samples.shape[0]
                    subject_shifts = shift + np.random.normal(0, shift_noise, n_subjects)
                    # Broadcast shift across other dimensions
                    shift_shape = [n_subjects] + [1] * (samples.ndim - 1)
                    applied_shift = subject_shifts.reshape(shift_shape)
                else:
                    applied_shift = shift + np.random.normal(0, shift_noise)
            else:
                applied_shift = shift
            
            samples += applied_shift
        
        return samples
    
    def variance(self, mu: np.ndarray, shape: float = 1.0, **kwargs) -> np.ndarray:
        return mu**2 / shape

class InverseGaussianFamily(DistributionFamily):
    """Inverse Gaussian distribution family
    
    Parameters
    ----------
    lambda : float, optional
        Dispersion parameter. If not provided, uses the canonical 
        parameterization lambda = mu^2 via utils.lsolve(mu).
        Higher values reduce variability.
    """
    
    def simulate(self, mu: np.ndarray, lambda_param: float = None, shift: float = 0.0,
                 shift_noise: float = 0.0, **kwargs) -> np.ndarray:
        """Generate samples from Inverse Gaussian distribution with optional shift
        
        Parameters
        ----------
        lambda_param : float, optional
            Dispersion parameter. If None, uses utils.lsolve(mu) = mu^2 
            which gives the canonical parameterization where mu^3/lambda = mu.
            Can also be passed as 'lambda' in family_params dictionary.
        shift : float, default=0.0
            Minimum response time shift (e.g., 200ms for motor response time)
        shift_noise : float, default=0.0
            Standard deviation of random noise added to shift for each subject
        """
        # Handle both 'lambda' and 'lambda_param' parameter names for backwards compatibility
        if lambda_param is None and 'lambda' in kwargs:
            lambda_param = kwargs['lambda']
        
        # Check for non-positive means
        if np.any(mu <= 0):
            n_negative = np.sum(mu <= 0)
            n_total = mu.size
            pct_negative = 100 * n_negative / n_total
            
            warnings.warn(
                f"InverseGaussianFamily.simulate: {n_negative}/{n_total} ({pct_negative:.1f}%) "
                f"mean values are non-positive (min={np.min(mu):.3f}). "
                f"This often occurs with identity link when random effects are large "
                f"relative to baseline. Consider: (1) increasing baseline values, "
                f"(2) reducing random effect SDs, or (3) using inverse link instead.",
                UserWarning
            )
            
            # Clip to small positive values to prevent crash
            mu = np.maximum(mu, 1e-6)
        
        # Use default lambda parameterization if not provided
        if lambda_param is None:
            lambda_param = utils.lsolve(mu)  # lambda = mu^2
        
        # Use numpy's wald distribution (proper parameterization)
        # numpy.random.wald(mean, scale) where scale = mean^3/lambda
        # This gives the correct mean and variance relationship
        samples = np.random.wald(mu, mu**3 / lambda_param)
        
        if shift > 0 or shift_noise > 0:
            # Apply shift with optional subject-level noise
            if shift_noise > 0:
                # Get unique subjects (assume first dimension is subjects)
                if samples.ndim >= 1:
                    n_subjects = samples.shape[0]
                    subject_shifts = shift + np.random.normal(0, shift_noise, n_subjects)
                    # Broadcast shift across other dimensions
                    shift_shape = [n_subjects] + [1] * (samples.ndim - 1)
                    applied_shift = subject_shifts.reshape(shift_shape)
                else:
                    applied_shift = shift + np.random.normal(0, shift_noise)
            else:
                applied_shift = shift
            
            samples += applied_shift
        
        return samples
    
    def variance(self, mu: np.ndarray, lambda_param: float = None, **kwargs) -> np.ndarray:
        """Variance function for Inverse Gaussian distribution
        
        Parameters
        ----------
        lambda_param : float, optional
            Dispersion parameter. If None, uses utils.lsolve(mu) = mu^2.
            Can also be passed as 'lambda' in family_params dictionary.
        """
        # Handle both 'lambda' and 'lambda_param' parameter names for backwards compatibility
        if lambda_param is None and 'lambda' in kwargs:
            lambda_param = kwargs['lambda']
            
        if lambda_param is None:
            lambda_param = utils.lsolve(mu)  # lambda = mu^2
        return mu**3 / lambda_param

class LogNormalFamily(DistributionFamily):
    """Log-Normal distribution family
    
    Parameters
    ----------
    sigma : float, default=1.0
        Standard deviation parameter on the log scale.
    """
    
    def simulate(self, mu: np.ndarray, sigma: float = 1.0, **kwargs) -> np.ndarray:
        # If mu is on log scale (from log link), use directly
        # Otherwise, need to convert
        if isinstance(self.link, LogLink):
            # mu is already log-scale
            return np.random.lognormal(mu, sigma, size=mu.shape)
        else:
            # Convert to log scale
            log_mu = np.log(np.maximum(mu, 1e-10))
            return np.random.lognormal(log_mu, sigma, size=mu.shape)
    
    def variance(self, mu: np.ndarray, sigma: float = 1.0, **kwargs) -> np.ndarray:
        if isinstance(self.link, LogLink):
            # mu is log-scale, convert to original scale for variance
            original_mu = np.exp(mu)
            return original_mu**2 * (np.exp(sigma**2) - 1)
        else:
            return mu**2 * (np.exp(sigma**2) - 1)

# Factory Functions
def get_link_function(link_name: str) -> LinkFunction:
    """Get link function instance by name
    
    Parameters
    ----------
    link_name : str
        Name of the link function: 'identity', 'log', 'inverse', 'sqrt'
        
    Returns
    -------
    LinkFunction
        Link function instance
        
    Raises
    ------
    ValueError
        If link_name is not recognized
    """
    link_functions = {
        'identity': IdentityLink(),
        'log': LogLink(),
        'inverse': InverseLink(),
        'sqrt': SqrtLink()
    }
    
    if link_name not in link_functions:
        raise ValueError(f"Unknown link function: {link_name}. "
                        f"Available: {list(link_functions.keys())}")
    
    return link_functions[link_name]


def get_family(family_name: str, link_name: str) -> DistributionFamily:
    """Get distribution family instance with specified link
    
    Parameters
    ----------
    family_name : str
        Name of the distribution family: 'gaussian', 'gamma', 'inverse_gaussian', 'lognormal'
    link_name : str
        Name of the link function: 'identity', 'log', 'inverse', 'sqrt'
        
    Returns
    -------
    DistributionFamily
        Distribution family instance with the specified link
        
    Raises
    ------
    ValueError
        If family_name is not recognized
    """
    link = get_link_function(link_name)
    
    families = {
        'gaussian': GaussianFamily(link),
        'gamma': GammaFamily(link),
        'inverse_gaussian': InverseGaussianFamily(link),
        'lognormal': LogNormalFamily(link)
    }
    
    if family_name not in families:
        raise ValueError(f"Unknown family: {family_name}. "
                        f"Available: {list(families.keys())}")
    
    return families[family_name]


def validate_family_link_combination(family_name: str, link_name: str) -> None:
    """Validate that family and link combination is supported
    
    Parameters
    ----------
    family_name : str
        Name of the distribution family
    link_name : str
        Name of the link function
        
    Raises
    ------
    ValueError
        If the combination is not valid or supported
    """
    if family_name not in VALID_FAMILY_LINK_COMBINATIONS:
        raise ValueError(f"Unknown family: {family_name}")
    
    if link_name not in VALID_FAMILY_LINK_COMBINATIONS[family_name]:
        raise ValueError(
            f"Invalid link '{link_name}' for family '{family_name}'. "
            f"Valid links: {VALID_FAMILY_LINK_COMBINATIONS[family_name]}"
        )


def get_available_families() -> List[str]:
    """Get list of available distribution families
    
    Returns
    -------
    List[str]
        List of available family names
    """
    return list(VALID_FAMILY_LINK_COMBINATIONS.keys())


def get_available_links(family_name: str = None) -> List[str]:
    """Get list of available link functions
    
    Parameters
    ----------
    family_name : str, optional
        If provided, return only links valid for this family
        
    Returns
    -------
    List[str]
        List of available link names
    """
    if family_name is None:
        # Return all available links
        all_links = set()
        for links in VALID_FAMILY_LINK_COMBINATIONS.values():
            all_links.update(links)
        return sorted(list(all_links))
    else:
        if family_name not in VALID_FAMILY_LINK_COMBINATIONS:
            raise ValueError(f"Unknown family: {family_name}")
        return VALID_FAMILY_LINK_COMBINATIONS[family_name]


def get_family_link_combinations() -> Dict[str, List[str]]:
    """Get all valid family/link combinations
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping family names to lists of valid link names
    """
    return VALID_FAMILY_LINK_COMBINATIONS.copy()


def create_family_summary() -> Dict[str, Dict[str, Any]]:
    """Create a summary of all available families and their properties
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Summary information for each family
    """
    return {
        'gaussian': {
            'description': 'Normal/Gaussian distribution for traditional linear models',
            'use_case': 'Linear mixed models, may produce negative values',
            'canonical_link': 'identity',
            'valid_links': get_available_links('gaussian'),
            'parameters': ['sigma'],
            'rt_suitability': 'Poor (allows negative values)',
            'typical_params': {'sigma': 100.0}
        },
        'gamma': {
            'description': 'Gamma distribution, excellent for right-skewed positive data',
            'use_case': 'Reaction time data, always positive, flexible shape',
            'canonical_link': 'inverse',
            'valid_links': get_available_links('gamma'),
            'parameters': ['shape'],
            'rt_suitability': 'Excellent (positive, right-skewed)',
            'typical_params': {'shape': 3.0}
        },
        'inverse_gaussian': {
            'description': 'Inverse Gaussian distribution with heavy right tail',
            'use_case': 'Reaction time data with occasional very slow responses',
            'canonical_link': 'inverse',
            'valid_links': get_available_links('inverse_gaussian'),
            'parameters': ['lambda'],
            'rt_suitability': 'Excellent (heavy tail, diffusion model basis)',
            'typical_params': {'lambda': None},  # Uses utils.lsolve(mu) = mu^2 by default
            'default_behavior': 'lambda = mu^2 (canonical parameterization)'
        },
        'lognormal': {
            'description': 'Log-Normal distribution, classic choice for RT data',
            'use_case': 'Multiplicative effects in reaction time',
            'canonical_link': 'log',
            'valid_links': get_available_links('lognormal'),
            'parameters': ['sigma'],
            'rt_suitability': 'Good (positive, multiplicative interpretation)',
            'typical_params': {'sigma': 0.2}
        }
    }
