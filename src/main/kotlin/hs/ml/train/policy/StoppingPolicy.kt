package hs.ml.train.policy

interface StoppingPolicy {
    fun reset()
    fun shouldStop(epoch: Int, loss: Double): Boolean
}