package hs.ml

import hs.ml.importer.CsvImporter
import hs.ml.importer.DataImporter
import hs.ml.importer.LinearDataGenerator
import hs.ml.loss.MeanSquaredError
import hs.ml.metric.RootMeanSquaredError
import hs.ml.model.LinearRegressor
import hs.ml.scaler.StandardScaler
import hs.ml.train.ModelFactory
import hs.ml.train.optimizer.SGD
import hs.ml.util.formatBytes
import java.io.File
import java.util.Scanner

fun main() {
    println("\n\n================================")
    println("OOP Machine Learning Project")
    println("PWD : ${File(".").canonicalFile}")
    println("CPU : ${Runtime.getRuntime().availableProcessors()} cores")
    println("Mem : ${formatBytes(Runtime.getRuntime().maxMemory())}")
    println()

    val importer = CsvImporter("data/housing.csv")

    println("데이터 불러오는 중...")
    val (x, y) = importer.read()
    println("데이터 불러오기 완료!")
    println("x: ${x.shape}, y: ${y.shape}")

    println("**모델 선택 단계**")

    val model = ModelFactory.create<LinearRegressor>()
        .setScaler(StandardScaler())
        .setLoss(MeanSquaredError())
        .setOptimizer(SGD(lr = 0.01))
        .addMetric(RootMeanSquaredError())
        .getModel()
}
