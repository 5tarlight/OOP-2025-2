package hs.ml.train.policy

class EarlyStoppingPolicy(private val patience: Int = 10, private val minDelta: Double = 1e-4) : StoppingPolicy {
    private var bestLoss = Double.MAX_VALUE
    private var badEpochs = 0

    override fun reset() {
        bestLoss = Double.MAX_VALUE
        badEpochs = 0
    }

    override fun shouldStop(epoch: Int, loss: Double): Boolean {
        if (loss.isNaN()) return false

        val isImprovement = loss < bestLoss - minDelta

        if (isImprovement) {
            bestLoss = loss
            badEpochs = 0
            return false
        }

        badEpochs++
        return badEpochs >= patience
    }
}