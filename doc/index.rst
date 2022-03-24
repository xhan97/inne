.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to inne's documentation!
============================================

inne is a pure python implementation for Isolation-based anomaly detection using nearest-neighbor ensembles based on the paper:

    Tharindu R., et al. `Isolation-based anomaly detection using nearest-neighbor ensembles. <https://onlinelibrary.wiley.com/doi/abs/10.1111/coin.12156>`__ Computational Intelligence (2018).

The first successful isolation-based anomaly detector, ie, iForest, uses
trees as a means to perform isolation. Although it has been shown to
have advantages over existing anomaly detectors, we have identified 4
weaknesses, ie, its inability to detect local anomalies, anomalies with
a high percentage of irrelevant attributes, anomalies that are masked by
axis-parallel clusters, and anomalies in multimodal data sets. To
overcome these weaknesses, this paper shows that an alternative
isolation mechanism is required and thus presents iNNE or isolation
using Nearest Neighbor Ensemble. Although relying on nearest neighbors,
iNNE runs significantly faster than the existing nearest neighbor-based
methods such as the local outlier factor, especially in data sets having
thousands of dimensions or millions of instances. This is because the
proposed method has linear time complexity and constant space
complexity.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   quick_start



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contribution

   contribution

`Getting started <install.html>`_
-------------------------------------

`API Documentation <api.html>`_
-------------------------------

`Examples <auto_examples/index.html>`_
--------------------------------------

`Contribution <contribution.html>`_
-------------------------------
