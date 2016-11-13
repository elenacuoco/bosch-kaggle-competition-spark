/*
 Copyright (c) 2014 by Contributors
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */


import java.util.Calendar

import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.feature._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.col


object PCAXGBoost {
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  def main(args: Array[String]): Unit = {

    val inputTrainPath = ".."
    val inputTestPath = inputTrainPath
    val outputModelPath = "/home/model-xgboost/"

    // create SparkSession
    val spark = SparkSession
      .builder()
      .appName("Bosch-PCAXGBoost-Spark")
      .config("spark.executor.memory", "4G")
      .config("spark.executor.cores", "7")
      .config("spark.driver.memory", "2G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.default.parallelism", "4")
      .master("local[4]")
      .getOrCreate()
    val now = Calendar.getInstance()
    val date = java.time.LocalDate.now
    val currentHour = now.get(Calendar.HOUR_OF_DAY)
    val currentMinute = now.get(Calendar.MINUTE)

    val direct = "./results/" + date + "-" + currentHour + "-" + currentMinute + "/"
    println(s"Saving results in $direct")


    ///read data from disk


    val dataset0 = spark.read.option("header", "true").option("inferSchema", true).csv(inputTrainPath + "/input/train_numeric.csv")
    val datatest0 = spark.read.option("header", "true").option("inferSchema", true).csv(inputTestPath + "/input/test_numeric.csv")
    //read "magic"features
   //val leak = spark.read.option("header", "true").option("inferSchema", true).csv(inputTestPath + "/input/leak.csv").select("Id", "LStartTime")
    val magic = spark.read.option("header", "true").option("inferSchema", true).csv(inputTestPath + "/input/magic.csv")
    val happy = spark.read.option("header", "true").option("inferSchema", true).csv(inputTestPath + "/input/happy.csv")
   // val Magicdata0 = leak.join(magic, leak("Id") === magic("Id")).drop(leak.col("Id"))
    val Magicdata = magic.join(happy, happy("Id") === magic("Id")).drop(happy.col("Id"))


    val dataset00 = dataset0.join(Magicdata, dataset0("Id") === Magicdata("Id")).drop(dataset0.col("Id"))
    val datatest00 = datatest0.join(Magicdata, datatest0("Id") === Magicdata("Id")).drop(datatest0.col("Id"))
    //df.join(otherDf).drop(otherDf.col("id"))
    val dateset = spark.read.option("header", "true").option("inferSchema", true).csv(inputTrainPath + "/input/train_date.csv").na.fill(9999999)
    val datetest = spark.read.option("header", "true").option("inferSchema", true).csv(inputTestPath + "/input/test_date.csv").na.fill(9999999)

    //datetest.printSchema()
    //prepare data for ML
    val headerDate = dateset.columns.filter(!_.contains("Id")).filter(!_.contains("L3_S46_D4135"))
    val assemblerDate = new VectorAssembler()
      .setInputCols(headerDate)
      .setOutputCol("IDatefeatures")

    val dateData = assemblerDate.transform(dateset).select("Id", "IDatefeatures")
    val dateDatatest = assemblerDate.transform(datetest).select("Id", "IDatefeatures")
    println("VectorAssembler Done!")

    val selector = new PCA()
      .setInputCol("IDatefeatures")
      .setOutputCol("Datefeatures")
      .setK(400)
    //dateData.printSchema()

    val selectorModel = selector.fit(dateData)
    val dateSel = selectorModel.transform(dateData).select("Id", "Datefeatures")
    val dateSeltest = selectorModel.transform(dateDatatest).select("Id", "Datefeatures")
    println("selector for Date Done!")

    val dataset= dataset00.join(dateSel, dataset00("Id") === dateSel("Id")).drop(dataset00.col("Id"))
    val datatest = datatest00.join(dateSeltest, datatest00("Id") === dateSeltest("Id")).drop(datatest00.col("Id"))


    //fill NA with 9999999
    println(dataset.count())
    val df1 = dataset.dropDuplicates(dataset.columns.filter(!_.contains("Id")).filter(!_.contains("Response"))) //.sample(true,0.4,10)

    val df = df1.na.fill(9999999)
    val df_test = datatest.na.fill(9999999)

    println(df.count())
    //prepare data for ML
    val header = df.columns.filter(!_.contains("Id")).filter(!_.contains("Response")).filter(!_.contains("Datefeatures"))

    val assemblerN = new VectorAssembler()
      .setInputCols(header)
      .setOutputCol("Nfeatures")


    val train_data0 = assemblerN.transform(df)
    val test_data0 = assemblerN.transform(df_test)
    println("Numeric VectorAssembler Done!")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Nfeatures", "Datefeatures"))
      .setOutputCol("features")

    train_data0.printSchema()
    val train_DF0 = assembler.transform(train_data0)
    val test_DF0 = assembler.transform(test_data0)
    println("VectorAssembler Done!")
    train_DF0.printSchema()



    val train0 = train_DF0.withColumn("label", df("Response").cast("double"))
    val test0 = test_DF0.withColumn("label", lit(1.0)).withColumnRenamed("Id", "id")

    val train1 = train0.select("label", "features")
    val testF = test0.select("id", "label", "features")
    train1.printSchema()
    testF.printSchema()
    println("Ready to go")
    // Split the data into training and test sets (20% held out for testing).

    val Array(trainingData, testData) = train1.randomSplit(Array(0.8, 0.2), seed = 7)
    //train1.show(4)
    //testF.show(4)
    // number of iterations
    val numRound = 1000
    val numWorkers = 4
    // training parameters

    val paramMap = List(
      "eta" -> 0.023f,
      "max_depth" -> 13,
      "min_child_weight" -> 3.5,
      "subsample" -> 0.88,
      "colsample_bytree" -> 0.77,
      "colsample_bylevel" -> 0.67,
      "base_score" -> 0.005,
      "eval_metric" -> "auc",
      "seed" -> 38,
      "silent" -> 1,
      "objective" -> "binary:logistic").toMap
    println("Starting Xgboost ")

    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingData, paramMap,
      round = numRound, nWorkers = numWorkers, useExternalMemory = false)
    val predictions = xgBoostModelWithDF.setExternalMemory(false).transform(testData)


    // DataFrames can be saved as Parquet files, maintaining the schema information
    predictions.select("label", "probabilities").write.parquet(direct + "preds.parquet")

    //prediction on test set for submission file
    val submission = xgBoostModelWithDF.setExternalMemory(false).transform(testF)

    submission.select("id", "probabilities").write.parquet(direct + "submission.parquet")

    spark.stop()
  }
}

