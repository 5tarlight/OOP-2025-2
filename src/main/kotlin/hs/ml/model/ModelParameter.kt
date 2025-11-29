package hs.ml.model

import hs.ml.loss.Loss
import hs.ml.metric.Metric
import hs.ml.scaler.Scaler
import hs.ml.train.optimizer.Optimizer

data class ModelParameter(
    var scaler: Scaler? = null,
    var loss: Loss? = null,
    var metric: MutableList<Metric> = mutableListOf(),
    var optimizer: Optimizer? = null
)
