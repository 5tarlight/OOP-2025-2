package hs.ml.train.optimizer

import hs.ml.autograd.Node

abstract class Optimizer {
    abstract val lr: Double

    abstract fun step(params: List<Node>)
}