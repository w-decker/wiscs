from .simulate import DataGenerator
import numpy as np
import numpy.typing as npt
import math

def make_tasks(low, high, n) -> npt.ArrayLike:
    """Generate task parameters"""
    return np.random.permutation(np.linspace(low, high, n).round(0))

def deltas(DG:DataGenerator, idx:str) -> npt.ArrayLike:
    """Calculate differences across experimental variable
    
    Parameter
    ---------
    idx:str
        Options: 'participant', 'trial', 'question'
    """
    d = []
    image = DG.data[0]
    word = DG.data[1]
    if idx == 'participant':
        for i in range(DG.params['n']['participant']):
            d.append(np.abs(image[i, :, :].mean() - word[i, :, :].mean()))
    elif idx == 'question':
        for i in range(DG.params['n']['question']):
            d.append(np.abs(image[:, i, :].mean() - word[:, i, :].mean()))
    elif idx == 'trial':
        for i in range(DG.params['n']['question']):
            d.append(np.abs(image[:, :, i].mean() - word[:, :, i].mean()))   
    return np.array(d)

def nearest_square_dims(n:int) -> int | int:
    """Reshape vector to nearest square that minimizes the difference between dimensions n x m"""
    rows = math.floor(math.sqrt(n))
    cols = math.ceil(math.sqrt(n))
    while rows * cols < n:
        if (cols - rows) <= 1:
            cols += 1
        else:
            rows += 1
    return rows, cols

def pairwise_deltas(DG: DataGenerator, idx: str) -> npt.NDArray:
    """
    Compute a grid of pairwise absolute differences for a given experimental variable
    """
    image = DG.data[0]
    word = DG.data[1]

    if idx == 'participant':
        i = np.array([image[i, :, :].mean() for i in range(DG.params['n']['participant'])])
        w = np.array([word[i, :, :].mean() for i in range(DG.params['n']['participant'])])
        return np.abs(i[:, np.newaxis] - w[np.newaxis, :])

    elif idx == 'question':
        i = np.array([image[:, i, :].mean() for i in range(DG.params['n']['question'])])
        w = np.array([word[:, i, :].mean() for i in range(DG.params['n']['question'])])
        return np.abs(i[:, np.newaxis] - w[np.newaxis, :])