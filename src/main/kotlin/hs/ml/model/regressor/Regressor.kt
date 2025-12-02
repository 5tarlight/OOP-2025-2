package hs.ml.model.regressor

import hs.ml.math.Tensor
import hs.ml.model.Model

abstract class Regressor: Model() {
    abstract fun predict(x: Tensor): Tensor
}