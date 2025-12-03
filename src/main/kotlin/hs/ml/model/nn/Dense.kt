package hs.ml.model.nn

import java.util.concurrent.ThreadLocalRandom
import hs.ml.autograd.Node
import hs.ml.math.Tensor
import kotlin.math.sqrt

class Dense(val inputSize: Int, val outputSize: Int) : Layer() {
    var weights = Node(Tensor(inputSize, outputSize) { _, _ ->
        ThreadLocalRandom.current().nextGaussian() * sqrt(2.0 / inputSize)
    })
        private set
    var bias = Node(Tensor(1, outputSize, 0.0))
        private set

    override fun forward(input: Node): Node {
        return (input * weights) + bias
    }

    override fun params(): List<Node> = listOf(weights, bias)
}