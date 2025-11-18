package hs.ml.model

import hs.ml.data.Tensor
import kotlin.math.ln

class Evaluator {
    companion object {
        fun mse(y: Tensor, yhat: Tensor): Double {
            require(y.row == yhat.row && y.col == yhat.col) {
                "y와 yhat의 크기가 일치하지 않습니다."
            }

            var sum = 0.0
            for (i in 0 until y.row) {
                for (j in 0 until y.col) {
                    val diff = y[i, j] - yhat[i, j]
                    sum += diff * diff
                }
            }

            return sum / (y.row * y.col)
        }

        fun binaryCrossEntropy(y: Tensor, yhat: Tensor): Double {
            require(y.row == yhat.row && y.col == yhat.col) {
                "y와 yhat의 크기가 일치하지 않습니다."
            }

            var sum = 0.0
            for (i in 0 until y.row) {
                for (j in 0 until y.col) {
                    val yi = y[i, j]
                    val yhati = yhat[i, j]
                    sum += - (yi * ln(yhati + 1e-15) + (1 - yi) * ln(1 - yhati + 1e-15))
                }
            }

            return sum / (y.row * y.col)
        }
    }
}
