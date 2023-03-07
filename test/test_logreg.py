"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import loadDataset, LogisticRegressor

def test_prediction():
    np.random.seed(0)
    x,y = loadDataset()

    lr = LogisticRegressor(6,max_iter=100, learning_rate=0.001)
    n = 1800
    lr.train_model(x[:n],y[:n], x[n:], y[n:])
    X_val = x[n:]
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    assert np.mean(np.around(lr.make_prediction(X_val)) == y[n:]) > 0.5

def test_loss_function():
    lr = LogisticRegressor(6,max_iter=100, learning_rate=0.001)
    
    assert lr.loss_function(np.array([0,1]),np.array([0,0])) > 0
    assert np.around(lr.loss_function(np.array([0,1]),np.array([0,1]))) == 0


def test_gradient():
    np.random.seed(0)
    lr = LogisticRegressor(2,max_iter=100, learning_rate=0.001)
    lr.W = np.array([0,0])
    g = lr.calculate_gradient(np.array([1]), np.array([[1,2]]))
    assert np.all(g == [-0.5, -1])

def test_training():
    np.random.seed(0)
    x,y = loadDataset()

    lr = LogisticRegressor(6,max_iter=100, learning_rate=0.001)
    n = 1800
    lr.train_model(x[:n],y[:n], x[n:], y[n:])
    X_val = x[n:]
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    assert lr.loss_hist_train[0]>lr.loss_hist_train[-1]