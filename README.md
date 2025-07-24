# wiscs
Simulating experimental data for the project, Word and Images in Shared Conceptual Space (WISCS). This code operates under the assumption that a single reaction time, $S$, is a combination of some cognitive processing, $C$, plus some noise. This is modeled below.

$$
S = C + \epsilon
$$

However, reaction times across subjects and factors must also be taken into account. Data can be modeled with random effects for various factors by providing an `re_formula` and correlation matrices. Further, reaction times are typically non-normal. Data are simulated with this in mind.

## Installation 

Install with `pip`:
```bash
pip install git+https://github.com/w-decker/wiscs.git
```

>[!WARNING]
> `wiscs` is _not_ compatible with Google Colab.

## Usage
`wiscs` is based on a set of parameters provided by the user. These parameters contain the necessary variables to set up an experiment. 

### Setting up `params`

```python
params = {
    'word.perceptual': ...,
    'image.perceptual': ...,

    'word.conceptual': ...,
    'image.conceptual': ...,

    'word.task': ...,
    'image.task': ...,

    'sd.item': ...,
    'sd.question': ...,
    'sd.subject': ...,
    'sd.error': ...,
    'sd.re_formula':...,
    'sd.shift':...,
    'sd.shift_noise':...,

    'corr.{factor}':...,

    'n.subject': ...,
    'n.question': ...,
    'n.item': ...,

    'glm.family': ..., 'glm.link': ..., 'glm.family_params': ...
}
```

The variables assigned to each parameter help build the necessary "environment" for simulating the data. These variables must be distributed to the necessary code internally, so one must set them.

```python
import wiscs
wiscs.set_params(params)
```
```
>>> Params set successfully
```
You can also grab a template `params` using the same function. It will return a dictionary which can be filled with the necessary values.

```python
template = wiscs.set_params(return_empty=True)
```
```
Params must be a dictionary with the following keys:
 dict_keys(['word.perceptual', 'image.perceptual', 'word.conceptual', 'image.conceptual', 'word.task', 'image.task', 'sd.item', 'sd.question', 'sd.subject', 'sd.modality', 'sd.re_formula', 'sd.error', 'sd.shift', 'sd.shift_noise', 'corr.subject', 'corr.question', 'corr.item', 'corr.modality', 'n.subject', 'n.question', 'n.item', 'glm.family', 'glm.link', 'glm.family_params'])

```
Printing `template` will tell you the expected types for each parameter. 

### Generating data

Data are generated using the `DataGenerator` class. If you have set the paramaeters, then this class will have access to them upon import.

```python
from wiscs.simulate import DataGenerator
DG = DataGenerator()
```

To generated the data, simply implement the `.fit_transform()` method.

```python
DG.fit_transform()
```

If you want to provide the `DataGenerator()` class with a new set of parameters, you can either reset them using `wiscs.set_params()` or you can simply provide a new set of params to the `.fit_transform()` method. 

```python
DG.fit_transform(params, overwrite=True)
```
Setting `overwrite=True` means that this new set of params will overwrite the original ones set using `wiscs.set_params()`. The default is `False`, which makes it easy to substitue any number of parameter dictionaries iterativelys. 

Data can be accessed with the `.data` attribute, which returns a `NamedTuple` with attributes for each modality

```python
word, image = DG.data
```

You can also extract summary statistics from your generated dataset.

```
DG.summary()
```

Data can also be converted to a `pandas` dataframe for easier use.

```python
df = DG.to_pandas()
```

## A note on how random effects are generated
Data generation begins with constructing a baseline matrix, which represents the expected reaction times before introducing variability. This matrix is built from predefined perceptual, conceptual, and task-related parameters, fully crossing subjects, questions, items, and modalities. Next, random effects are introduced to account for variability across subjects, questions, and items (if the user wishes). These effects are drawn from multivariate normal distributions, where correlation structures among effects are preserved using Cholesky decomposition (or eigen decomposition). Cholesky decomposition factorizes the covariance matrix into a lower triangular matrix, which allows us to efficiently generate correlated random deviations by multiplying it with standard normal samples. Structured deviations are then added to the baseline matrix, along with residual noise, to produce the final dataset, ensuring that both systematic and random variability reflect realistic experimental conditions.

## Contributing
Please submit issues or PRs with detailed information. Feel free to contact me [here](mailto:will.decker@gatech.edu?subject=wiscs).