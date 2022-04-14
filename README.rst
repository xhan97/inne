.. -*- mode: rst -*-

|PyPI|_ |ReadTheDocs|_ |Downloads|_ |GitHubCI|_  |Codecov|_ |CircleCI|_ 


.. |GitHubCI| image:: https://github.com/xhan97/inne/actions/workflows/inne-CI.yml/badge.svg
.. _GithubCI: https://github.com/xhan97/inne/actions/workflows/inne-CI/

.. |PyPI| image:: https://badge.fury.io/py/inne.svg
.. _PyPI: https://badge.fury.io/py/inne

.. |Codecov| image:: https://codecov.io/gh/xhan97/inne/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/xhan97/inne

.. |CircleCI| image:: https://circleci.com/gh/xhan97/inne.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/xhan97/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/inne/badge/?version=latest
.. _ReadTheDocs: https://inne.readthedocs.io/en/latest/?badge=latest

.. |Downloads| image:: https://pepy.tech/badge/inne
.. _Downloads: https://pepy.tech/project/inne


iNNE
======================================================================

iNNE - Isolation-based anomaly detection using nearest-neighbor ensembles.

Based on the paper:

    Tharindu R., et al. `Isolation-based anomaly detection using nearest-neighbor ensembles. <https://onlinelibrary.wiley.com/doi/abs/10.1111/coin.12156>`__ Computational Intelligence (2018)

Matlab code of iNNE:

    https://github.com/zhuye88/iNNE

Introduction to the paper:

    https://www.jianshu.com/p/379a5898beb6

Abstract of the paper:

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

Documentation, including tutorials, are available on ReadTheDocs at
https://inne.readthedocs.io.

----------
Installing
----------

PyPI install, presuming you have an up to date pip.

.. code:: bash

   pip install inne

For a manual install of the latest code directly from GitHub:

.. code:: bash

    pip install git+https://github.com/xhan97/inne.git


Alternatively download the package, install requirements, and manually run the installer:

.. code:: bash

    wget https://codeload.github.com/xhan97/inne/zip/refs/heads/master
    unzip inne-master.zip
    rm inne-master.zip
    cd inne-master

    pip install -r requirements.txt

    python setup.py install

------------------
How to use iNNE
------------------

The inne package inherits from sklearn classes, and thus drops in neatly
next to other sklearn  with an identical calling API. Similarly it
supports input in a variety of formats: an array (or pandas dataframe) of shape ``(num_samples x num_features)``.

.. code:: python

    from inne import IsolationNNE
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(1000)

    clf = IsolationNNE(n_estimators=200, max_samples=16)
    clf.fit(data)
    anomaly_labels = clf.predict(data)

-----------------
Running the Tests
-----------------

The package tests can be run after installation using the command:

.. code:: bash

    pip install pytest 

or, if ``pytest`` is installed:

.. code:: bash

    pytest  inne/tests

If one or more of the tests fail, please report a bug at https://github.com/xhan97/inne/issues

--------------
Python Version
--------------

Python 3  is recommend  the better option if it is available to you.

------
Citing
------

If you have used this codebase in a scientific publication and wish to
cite it, please use the following publication (Bibtex format):

.. code:: bibtex

    @article{bandaragoda2018isolation,
            title={Isolation-based anomaly detection using nearest-neighbor ensembles},
            author={Bandaragoda, Tharindu R and Ting, Kai Ming and Albrecht, David and Liu, Fei Tony and Zhu, Ye and Wells, Jonathan R},
            journal={Computational Intelligence},
            volume={34},
            number={4},
            pages={968-998},
            year={2018},
            publisher={Wiley Online Library} }

------------------
How to contribute
------------------

Thanks for your interest in contributing to inne. A guide is shown in inne's `Documentation <https://inne.readthedocs.io/en/latest/contribution.html>`__.


License
-------

BSD license
