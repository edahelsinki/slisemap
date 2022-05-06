"""
__SLISEMAP: Combining local explanations with supervised dimensionality reduction___

SLISEMAP is a supervised dimensionality reduction method, that takes data, in the form of
vectors, and predictions from a "black box" regression or classification model as input.
SLISEMAP then simultaneously finds local explanations for all data items and builds a
(typically) two-dimensional global visualisation of the black box model such that data
items with similar local explanations are projected nearby. The explanations consists of
"white box" models that locally approximate the "black box" model.

SLISEMAP uses *PyTorch* for efficient optimisation, and optional GPU-acceleration. For
more information see the the repository (https://github.com/edahelsinki/slisemap) or
the paper (https://arxiv.org/abs/2201.04455).

__Citation__
> Björklund, A., Mäkelä, J. & Puolamäki, K. (2022).
> SLISEMAP: Supervised dimensionality reduction through local explanations.
> arXiv:2201.04455 [cs], https://arxiv.org/abs/2201.04455.
"""

from slisemap.slisemap import Slisemap
