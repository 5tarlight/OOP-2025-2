package hs.ml.preprocessing

import hs.ml.data.DataBatch
import hs.ml.preprocessing.policy.MissingPolicy

class DataPreprocessor(
    private val missingPolicy: MissingPolicy
){
    fun process(batch: DataBatch):DataBatch{
        println("DataPreprocessor: 전처리 시작...")
        val processedInputs = missingPolicy.handle(batch.inputs)
        val processedBatch = DataBatch(
            inputs = processedInputs,
            labels = batch.labels
        )
        println("DataPreprocessor: 전처리 완료.")
        return processedBatch
    }
}