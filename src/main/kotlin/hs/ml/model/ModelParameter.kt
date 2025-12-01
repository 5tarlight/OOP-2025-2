package hs.ml.model

import hs.ml.loss.Loss
import hs.ml.loss.MeanSquaredError
import hs.ml.metric.Metric
import hs.ml.train.optimizer.Optimizer
import hs.ml.train.optimizer.SGD

data class ModelParameter(
    var loss: Loss = MeanSquaredError(),
    var metric: MutableList<Metric> = mutableListOf(),
    var optimizer: Optimizer = SGD()
)
