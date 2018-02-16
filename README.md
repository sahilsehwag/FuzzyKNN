# Overview
This is a python implementation of Fuzzy KNN Algorithm. For more details or in-depth explanation look at this research paper *[A Fuzzy K-nearest Neighbor Algorithm](http://ieeexplore.ieee.org/document/6313426/)* by J.M Keller, M.R. Grey and J.A. Givens.

The main Fuzzy-KNN algorithm is implemented as a class named **_FuzzyKNN_**, which resides in fknn.py.
This class is implemented using scikit-learn's API. 
It inherits from BaseEstimator, ClassifierMixin and follows sklearn guidelines which allows it to be used as regular sklearn *Estimator*, making it useful as it can be used with sklearn's API.

A Jupyter Python Notebook is provided which also contains the implementation of FuzzyKNN for experimentation purposes. In this notebook accuracy of sklearn's **KNeighborsClassifier** is compared with FuzzyKNN on toy datasets like IRIS and Breast Cancer.

If you like this repo, look at this [machine-learning-algorithms](https://github.com/sahilsehwag/machine-learning-algorithms), where I implement various machine-learning algorithms as sklearn **Estimators**, and compares the accuracy of our custom implementation with sklearn's inbuilt implementations. Our custom implementations are commented for tutorial purposes, along with mathematics behind these algorithms.

# Dependencies
* [pandas](http://pandas.pydata.org/pandas-docs/stable/)
* [numpy](https://docs.scipy.org/doc/numpy/reference/index.html)
* [scikit-learn](http://scikit-learn.org/stable/documentation.html)
* [matplotlib](https://matplotlib.org/)

# Similar Repositories
* [SentiCircle Algorithm](https://github.com/sahilsehwag/Senticircle-Implementation)
* [Machine Learning Algorithms](https://github.com/sahilsehwag/machine-learning-algorithms)
* [Python Implementation of Various Data Structures](https://github.com/sahilsehwag/data-structures-python)
* [Python Implementation of Various Algorithms](https://github.com/sahilsehwag/algorithms-python)
