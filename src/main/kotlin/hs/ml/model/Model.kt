package hs.ml.model

import hs.ml.math.Tensor

abstract class Model {
    protected abstract var weights: Tensor
    protected abstract var bias: Double
    var param: ModelParameter = ModelParameter()
    var isTrained: Boolean = false

    abstract fun forward(x: Tensor): Tensor
}
