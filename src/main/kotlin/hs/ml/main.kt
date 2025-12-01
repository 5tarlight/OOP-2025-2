package hs.ml

import hs.ml.data.DataPipeline
import hs.ml.importer.CsvImporter
import hs.ml.loss.MeanSquaredError
import hs.ml.metric.RootMeanSquaredError
import hs.ml.model.LinearRegressor
import hs.ml.preprocessing.DataPreprocessor
import hs.ml.preprocessing.policy.ReplaceToAvgPolicy
import hs.ml.scaler.StandardScaler
import hs.ml.train.ModelFactory
import hs.ml.train.optimizer.SGD
import hs.ml.util.formatBytes
import java.io.File

fun main() {
    println("\n\n================================")
    println("OOP Machine Learning Project")
    println("PWD : ${File(".").canonicalFile}")
    println("CPU : ${Runtime.getRuntime().availableProcessors()} cores")
    println("Mem : ${formatBytes(Runtime.getRuntime().maxMemory())}")
    println("================================\n\n")
    println()

    println("**데이터 불러오기 및 전처리 단계**\n")
    println("데이터 불러오는 중...")

    val importer = CsvImporter("data/housing.csv")
    val pipeline = DataPipeline(
        importer = importer,
        preprocessor = DataPreprocessor(missingPolicy = ReplaceToAvgPolicy())
    )
    val (x, y) = pipeline.run()

    println("데이터 불러오기 완료!")
    println("x: ${x.shape}, y: ${y.shape}")

    println("\n**모델 선택 단계**\n")

    val model = ModelFactory.create<LinearRegressor>()
        .setScaler(StandardScaler())
        .setLoss(MeanSquaredError())
        .setOptimizer(SGD(lr = 0.01))
        .addMetric(RootMeanSquaredError())
        .getModel()

    println("모델 생성 완료! : $model")
}
