package hs.ml.model.nn

import hs.ml.autograd.Node

abstract class Layer {
    abstract fun forward(input: Node): Node
    open fun params(): List<Node> = emptyList()
}