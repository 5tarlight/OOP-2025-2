package hs.ml.autograd

import hs.ml.math.Tensor

//Computational Graph
class Node(var data: Tensor, val children: List<Node> = emptyList(), val debug: String = "") {
    var grad: Tensor = Tensor(data.row, data.col, 0.0)
    internal var _backward: () -> Unit = {}

    operator fun plus(other: Node): Node {
        val out = Node(this.data + other.data, listOf(this, other), "+")

        out._backward = {
            if (this.data.shape == out.grad.shape) {
                this.grad = this.grad + out.grad
            } else if (this.data.row == 1 && this.data.col == out.grad.col) {
                this.grad = this.grad + out.grad.sum(axis = 0)
            }

            if (other.data.shape == out.grad.shape) {
                other.grad = other.grad + out.grad
            } else if (other.data.row == 1 && other.data.col == out.grad.col) {
                other.grad = other.grad + out.grad.sum(axis = 0)
            }
        }
        return out
    }

    operator fun unaryMinus(): Node {
        return this * Node(Tensor(data.row, data.col, -1.0))
    }

    operator fun times(other: Node): Node {
        val out = Node(this.data * other.data, listOf(this, other), "*")

        out._backward = {
            this.grad = this.grad + (out.grad * other.data.T)
            other.grad = other.grad + (this.data.T * out.grad)
        }
        return out
    }

    fun map(transform: (Double) -> Double, derivative: (Double) -> Double): Node {
        val outData = this.data.map(transform)
        val out = Node(outData, listOf(this), "map")

        out._backward = {
            val localGrad = this.data.map(derivative)
            this.grad = this.grad + (out.grad.hadamard(localGrad))
        }
        return out
    }

    fun pow(exponent: Int): Node {
        val out = Node(this.data.pow(exponent), listOf(this), "^$exponent")

        out._backward = {
            val n = exponent.toDouble()
            val localDerivative = this.data.pow(exponent - 1) * n
            this.grad = this.grad + (out.grad.hadamard(localDerivative))
        }
        return out
    }

    fun backward(initialGrad: Tensor? = null) {
        val topo = mutableListOf<Node>()
        val visited = mutableSetOf<Node>()

        fun visit(v: Node) {
            if (v !in visited) {
                visited.add(v)
                for (child in v.children) visit(child)
                topo.add(v)
            }
        }
        visit(this)

        if (initialGrad != null) {
            this.grad = initialGrad
        } else {
            this.grad = Tensor(data.row, data.col, 1.0)
        }

        for (node in topo.reversed()) {
            node._backward()
        }
    }
}