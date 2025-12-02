package hs.ml.model.classifier

import hs.ml.math.Tensor
import hs.ml.model.Model

abstract class Classifier: Model() {
    abstract fun classify(x: Tensor, threshold: Double = 0.5): Tensor
}