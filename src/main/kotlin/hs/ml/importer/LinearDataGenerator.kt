package hs.ml.importer

import hs.ml.data.Tensor

class LinearDataGenerator(val coef: Double, val bias: Double, val noise: Double): DataImporter {
    override fun read(): Tensor {
        TODO("Not yet implemented")
    }

    override fun available(): Boolean = true
}
