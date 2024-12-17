# wiscs
Simulating experimental data for the project, Word and Images in Shared Conceptual Space (WISCS). This code operates under the assumption that a reaction time, $S$, is a combination of some cognitive processing, $C$, plus some noise. This is modeled below.

$$
S = C + \epsilon
$$

## Installation 

Install with `pip`:
```bash
pip install git+https://github.com/w-decker/wiscs.git
```

## Usage
`wiscs` is based on a set of parameters provided by the user. These parameters contain the necessary variables to set up an experiment. 

### Setting up `params`

```python
params = {
    'word.concept': ...,
    'image.concept': ...,
    'word.task': ...,
    'image.task': ...,

    'var.image': ...,
    'var.word': ...,
    'var.question': ...,
    'var.participant': ...,

    'n.participant': ...,
    'n.question': ...,
    'n.trial': ...
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
>>> Params must be a dictionary with the following keys:
 dict_keys(['word.concept', 'image.concept', 'word.task', 'image.task', 'var.image', 'var.word', 'var.question', 'var.participant', 'n.participant', 'n.question', 'n.trial'])
```
Printing `template` will tell you the expected types for each parameter. 

### Generating data

Data are generated using the `DataGenerator` class. If you have set the paramaeters, then this class will have access to them upon import.

```python
from wiscs.simulate import DataGenerator
DG = DataGenerator()
```

To generated the data, simply implement the `.fit()` method.

```python
DG.fit()
```

Data can be accessed with the `.data` attribute.

```python
image, word = DG.data
```

Data can also be converted to a `pandas` dataframe for easier use.

```python
df = DG.to_pandas()
```

### Plotting
You can also plot the data with some default plotting functionality, including some histograms, line plots, heatmaps and scatter plots. These generally require one or more instances of a `DataGenerator` class as an input argument. 

```python
from wiscs.plotting import Plot
P = Plot(DG)
P.grid(idx='question') # must index on an experimental variable
```

For more plotting details, see the notebook [here](https://github.com/w-decker/wiscs-simulation/blob/main/wiscs-simulations.ipynb).

## Contributing
Please submit issues or PRs with detailed information. Feel free to contact me [here](mailto:will.decker@gatech.edu?subject=wiscs).