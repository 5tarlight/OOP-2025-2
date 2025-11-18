package hs.ml.model

import hs.ml.data.Tensor

interface Model {
    fun fit(x: Tensor, y: Tensor, epochs: Int, lr: Double)
    fun predict(x: Tensor): Tensor

    fun evaluate(x: Tensor, y: Tensor, metric: (Tensor, Tensor) -> Double): Double {
        val yhat = predict(x)
        return metric(y, yhat)
    }
}
