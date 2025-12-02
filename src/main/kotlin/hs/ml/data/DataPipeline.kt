package hs.ml.data

import hs.ml.importer.DataImporter
import hs.ml.preprocessing.DataPreprocessor

class DataPipeline(
    private val importer: DataImporter,
    private val preprocessor: DataPreprocessor
) {
    fun run(): DataBatch {
        val rawBatch = importer.read()
        val processedBatch = preprocessor.process(rawBatch)
        return processedBatch
    }
}