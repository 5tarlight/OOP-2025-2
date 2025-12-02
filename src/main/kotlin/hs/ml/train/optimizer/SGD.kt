package hs.ml.train.optimizer

import hs.ml.autograd.Node

class SGD : Optimizer {
    override var lr: Double
        private set

    constructor(lr: Double = 0.01) {
        this.lr = lr
    }

    override fun step(params: List<Node>) {
        for (param in params) {
            val stepSize = param.grad * lr
            val newWeight = param.data - stepSize
            param.data = newWeight
        }
    }
}