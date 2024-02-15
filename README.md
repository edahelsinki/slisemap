|*This branch is frozen at version 1.6.0 and contains the experiments (and results) for the Slipmap paper in the [experiments](experiments/) directory!*|
|---|

# SLISEMAP: Combine supervised dimensionality reduction with local explanations

SLISEMAP is a supervised dimensionality reduction method, that takes data, in the form of vectors, and predictions from a "black box" regression or classification model as input. SLISEMAP then simultaneously finds local explanations for all data items and builds a (typically) two-dimensional global visualisation of the black box model such that data items with similar local explanations are projected nearby. The explanations consists of interpretable models that locally approximate the "black box" model.

SLISEMAP is implemented in *Python* using *PyTorch* for efficient optimisation, and optional GPU-acceleration. For more information see the [papers](#citations), the [examples](https://github.com/edahelsinki/slisemap/tree/main/examples), or the [documentation](https://edahelsinki.github.io/slisemap/slisemap).

*This library also includes the faster SLIPMAP variant, that uses "prototypes" to speed up
the calculations (linear time and memory complexity instead of quadratic).
SLIPMAP is largely compatible with SLISEMAP, just change the class name (`Slisemap` to `Slipmap`, see example below).*


## Citations

The full SLISEMAP paper ([arXiv](https://arxiv.org/abs/2201.04455) and [supplements](https://github.com/edahelsinki/slisemap/tree/slisemap_experiments)):
> *Björklund, A., Mäkelä, J., & Puolamäki, K. (2023).*  
> **SLISEMAP: Supervised dimensionality reduction through local explanations.**  
> Machine Learning 112, 1-43. DOI: [10.1007/s10994-022-06261-1](https://doi.org/10.1007/s10994-022-06261-1)  

The short demo paper ([video](https://youtu.be/zvcFYItwRlQ) and [slides](https://github.com/edahelsinki/slisemap/blob/main/examples/demo_presentation.pdf)):
> *Björklund, A., Mäkelä, J., & Puolamäki, K. (2023).*  
> **SLISEMAP: Combining Supervised Dimensionality Reduction with Local Explanations.**  
> Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2022. Lecture Notes in Computer Science, vol 13718. DOI: [10.1007/978-3-031-26422-1_41](https://doi.org/10.1007/978-3-031-26422-1_41).

The new SLIPMAP paper ([supplements](https://github.com/edahelsinki/slisemap/tree/slipmap_experiments)):
> *Björklund, A., Seppäläinen, L., & Puolamäki, K. (2024).*  
> **SLIPMAP: Fast and Robust Manifold Visualisation for Explainable AI**  
> To appear in: Advances in Intelligent Data Analysis XXII. IDA 2024. Lecture Notes in Computer Science.  


## Installation

To install the package just run:

```sh
pip install slisemap
```

Or install the latest version directly from [GitHub](https://github.com/edahelsinki/slisemap):

```sh
pip install git+https://github.com/edahelsinki/slisemap
```

### PyTorch

Since SLISEMAP utilises PyTorch for efficient calculations, you might want to install a version that is optimised for your hardware. See [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for details.


## Example

```python
import numpy as np
from slisemap import Slisemap

X = np.array(...)
y = np.array(...)
sm = Slisemap(X, y, radius=3.5, lasso=0.01)
sm.optimise()
sm.plot(clusters=5, bars=5)
```
![Example plot of the results from using SLISEMAP on the *Auto MPG* dataset](docs/autompg.webp)

To use the faster SLIPMAP variant just replace the relevant lines:

```python
from slisemap import Slipmap
sm = Slipmap(X, y, radius=2.0, lasso=0.01)
```

See the [examples](https://github.com/edahelsinki/slisemap/tree/main/examples) for more detailed examples, and the [documentation](https://edahelsinki.github.io/slisemap/slisemap.html) for more detailed instructions.
