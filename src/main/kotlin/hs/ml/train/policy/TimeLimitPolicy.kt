package hs.ml.train.policy

class TimeLimitPolicy(private val limitSeconds: Long) : StoppingPolicy {
    private var startTime: Long = 0

    override fun reset() {
        startTime = System.currentTimeMillis()
    }

    override fun shouldStop(epoch: Int, loss: Double): Boolean {
        val elapsedSeconds = (System.currentTimeMillis() - startTime) / 1000
        return elapsedSeconds >= limitSeconds
    }
}