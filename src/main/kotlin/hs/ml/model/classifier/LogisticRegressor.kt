package hs.ml.model.classifier

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.Dense

class LogisticRegressor : Classifier() {
    private var layer: Dense? = null
    val weights: Tensor
        get() = layer?.weights?.data ?: Tensor(0, 0)

    val bias: Tensor
        get() = layer?.bias?.data ?: Tensor(0, 0)

    override fun forward(x: Node): Node {
        if (layer == null) {
            layer = Dense(x.data.col, 1)
        }

        return layer!!.forward(x)
    }

    override fun params(): List<Node> {
        return layer?.params() ?: emptyList()
    }

    override fun classify(x: Tensor, threshold: Double): Tensor {
        val inputVal = Node(x)
        val outVal = forward(inputVal)
        return outVal.data
    }

    override fun toString(): String = "LogisticRegressor(weights=$weights, bias=$bias, isTrained=$isTrained)"
}