import numpy as np
import pytest

from keras import objectives
from keras import backend as K

from nasdaqpredictor.model import Model

_EPSILON = K.epsilon()


def custom_loss(y_true, y_pred):
    new_true = (1. - y_true) * 100 + 1.
    return 0


def test_loss():
    y = generate_vector()
    y_pred = generate_vector()

    ret = K.eval(custom_loss(y, y_pred))
    print(ret)
    assert ret.all() > 0
    assert y.shape == y_pred.shape


def generate_vector():
    return np.random.randint(0, 3, (10, 1)).astype(np.float64)
