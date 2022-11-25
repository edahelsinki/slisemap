"""
SLISEMAP: Combine local explanations with supervised dimensionality reduction
===============================================================================

SLISEMAP is a supervised dimensionality reduction method, that takes data, in the form of
vectors, and predictions from a "black box" regression or classification model as input.
SLISEMAP then simultaneously finds local explanations for all data items and builds a
(typically) two-dimensional global visualisation of the black box model such that data
items with similar local explanations are projected nearby. The explanations consists of
"white box" models that locally approximate the "black box" model.

SLISEMAP uses *PyTorch* for efficient optimisation, and optional GPU-acceleration.
For more information see the the [repository](https://github.com/edahelsinki/slisemap),
the [documentation](https://edahelsinki.github.io/slisemap/slisemap), or the
[paper](https://doi.org/10.1007/s10994-022-06261-1).

Citation
--------
> Björklund, A., Mäkelä, J. & Puolamäki, K. (2022).  
> SLISEMAP: Supervised dimensionality reduction through local explanations.  
> Machine Learning, DOI 10.1007/s10994-022-06261-1.  


Example Usage
-------------

    from slisemap import Slisemap
    import numpy as np

    X = np.array([[0.1,0.5,0.7], [0.8,0.9,1], [0.8,0.5,0.3], [0.1,0.2,0.3], [1,2,5], [2,3,4], [2,0,1]])
    y = np.array([1, 2, 3, 4, 1.5, 1.8, 1.7])
    sm = Slisemap(X, y, radius=3.5, lasso=1e-4, ridge=2e-4)
    sm.optimise()
    sm.plot()
"""

from slisemap.slisemap import Slisemap
