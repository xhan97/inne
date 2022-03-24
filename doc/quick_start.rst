#####################################
Quickstart
#####################################

Our goal here is to help you to get the first practical experience with inne and give you a brief overview on some basic functionalities of inne.

Inne
===================================================
The inne package inherits from sklearn classes, and thus drops in neatly next to other sklearn with an identical calling API. Similarly it supports input in a variety of formats: an array (or pandas dataframe) of shape (num_samples x num_features).

    >>> from inne import IsolationNNE
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [0, 0], [-20, 50], [3, 5]])
    >>> clf = IsolationNNE(n_estimators=200, max_samples=16)
    >>> clf.fit(X)
    >>> clf.predict(X)

Examples

* See `IsolationNNE example <auto_examples/plot_inne.html>`_  for an illustration of the use of IsolaitonNNE.
* See `Comparing anomaly detection algorithms <auto_examples/plot_anomaly_comparison.html>`_  for outlier detection on toy datasets for a comparison of ensemble.IsolationForest with neighbors.LocalOutlierFactor, svm.OneClassSVM (tuned to perform like an outlier detection method), linear_model.SGDOneClassSVM, and a covariance-based outlier detection with covariance.EllipticEnvelope.

