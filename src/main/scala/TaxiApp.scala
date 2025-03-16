import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import java.time.{DayOfWeek, LocalDate, LocalDateTime, ZoneId}
import java.time.format.{DateTimeFormatter, DateTimeFormatterBuilder}

object SparkApp {

  //it will be used to convert vector into scalar based on a condition
  val vectorToScalarUDF = udf((v: Vector) => if (v != null && v.size > 0) v(0) else 0.0)

  //all scalers that will be used
  //scaler for day
  var scaler: MinMaxScaler = _
  var scalerModel: MinMaxScalerModel = _
  //scaler for month
  var scalerMonth: MinMaxScaler = _
  var scalerModelMonth: MinMaxScalerModel = _
  //scaler for year
  var scalerYear: MinMaxScaler = _
  var scalerModelYear: MinMaxScalerModel = _

  //pipeline model that will be used for one hot encoding of location id
  var pipelineModel: PipelineModel = _

  //creating spark app
  val spark = SparkSession.builder
    .appName("MySparkApp")
    .master("local[*]")
    .getOrCreate()

  def main(args: Array[String]): Unit = {


    //limiting log messges to WARN
    spark.sparkContext.setLogLevel("WARN")

    //files that will be used
    val filePaths1 = (1 to 12).map(month => f"Data/yellow_tripdata_2024-$month%02d.parquet").toArray
    val filePaths2 = (1 to 12).map(month => f"Data/yellow_tripdata_2023-$month%02d.parquet").toArray
    val filePaths3 = (1 to 12).map(month => f"Data/yellow_tripdata_2022-$month%02d.parquet").toArray

    val filePaths4 = Array("Data/yellow_tripdata_2024-02.parquet")
    val df = loadData(spark, filePaths4)

    //val df = loadData(spark, filePaths1 ++ filePaths2 ++filePaths3)


    //all methods will be called here
    val processedDf = df
      .transform(preprocessData)
      .persist() //to avoid reloading/recomputing raw data (it cashes the data so that it doesnt need to be recomputed several times)
      .transform(groupTimeSlots)
      .persist()
      .transform(encodeTimeSlots)
      .transform(addRedDays)
      .transform(normalizeData)

    //saving training, validation and test data to use it in thr model
    val (trainingData, validationData, testData) = splitData(processedDf)

    //building the model
    val linRegModelBuilt = linRegModel(trainingData, validationData, testData)

    //calling the model with new data
    predictDemand("2024-02-01", "16:53:09", "13", linRegModelBuilt)

    //inspecting data, it is commented as it takes lots of time and it is not necessary every time
    //inspectData(processedDf)

    spark.stop()
  }


  //leading data, also specifying columns to lead
  def loadData(spark: SparkSession, paths: Array[String]): DataFrame = {
    val dfs = paths.map { path =>
      spark.read
        .parquet(path)
        .select(
          col("tpep_pickup_datetime"),
          col("PULocationID").cast("bigint").alias("PULocationID")
        )
    }
    dfs.reduce(_ union _)
  }


  //starting preprocessing by extracting data from features that are imported
  def preprocessData(df: DataFrame): DataFrame = {
    df.withColumn("day", dayofmonth(col("tpep_pickup_datetime")))
      .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))
      .withColumn("month", month(col("tpep_pickup_datetime")))
      .withColumn("year", year(col("tpep_pickup_datetime")))
      .withColumn("hour", hour(col("tpep_pickup_datetime")))
      .withColumn("minute", minute(col("tpep_pickup_datetime")))
      .withColumn("date", to_date(col("tpep_pickup_datetime")))
      .withColumn("time", date_format(col("tpep_pickup_datetime"), "HH:mm:ss"))
      .drop("tpep_pickup_datetime")
  }


  //time column will be grouped in 10 min time slots and adding a column demand that keeps record of how many rows are in each time slot
  def groupTimeSlots(df: DataFrame): DataFrame = {
    val dfWithTimeSlots = df.withColumn("time_slot",
      expr("floor(minute(time) / 10) * 10")
    ).withColumn("time_slot",
      concat_ws(":", date_format(col("time"), "HH"), lpad(col("time_slot").cast("string"), 2, "0"))
    )
    dfWithTimeSlots.groupBy("PULocationID", "day", "month", "year", "day_of_week", "hour", "time_slot", "date", "time")
      .agg(count("*").alias("demand"))
      .orderBy("PULocationID", "time_slot", "date")
  }


  //timeslots will be encoded in a way that it shows the minut of that day, with rule hh*60 +mm
  def encodeTimeSlots(df: DataFrame): DataFrame = {
    val dfWithTimeSlots = df.withColumn("time_slot_minutes",
      expr("hour(time_slot) * 60 + minute(time_slot)")
    ).drop("time_slot", "time")
    dfWithTimeSlots
  }


  //checking if a date is a red date and adding a new column 1 for yes, 0 for no
  def addRedDays(dataFrame: DataFrame): DataFrame = {
    //all red days
    val redDays2024 = Set(
      "2024-01-01", "2024-01-15", "2024-02-12", "2024-02-19",
      "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
      "2024-10-14", "2024-11-05", "2024-11-11", "2024-11-28", "2024-12-25"
    )
    val redDays2022 = Set(
      "2022-01-01", "2022-01-17", "2022-02-21", "2022-05-30",
      "2022-06-19", "2022-07-04", "2022-09-05", "2022-10-10",
      "2022-11-11", "2022-11-24", "2022-12-25"
    )
    val redDays2023 = Set(
      "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29",
      "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09",
      "2023-11-11", "2023-11-23", "2023-12-25"
    )
    //checking, adding and droping the column
    val allRedDays = redDays2024 ++ redDays2022 ++ redDays2023
    val isRedDay = udf((date: String) => if (allRedDays.contains(date)) 1 else 0)
    dataFrame.withColumn("red_date", isRedDay(col("date")))
      .drop("date")
  }


    //normalizing day into range 0 to 1. it is using minmax scaler
    //scaler model is initialized at the begining of the program
    //in this method it is saved
    def normalizeDay(dataFrame: DataFrame): DataFrame = {
      ///7day column needs to be transformed into vector format for minmaxscaller
      val assemblerDay = new VectorAssembler().setInputCols(Array("day")).setOutputCol("day_vec")
      val assembledDayDf = assemblerDay.transform(dataFrame)
      //scaling data
      if (scaler == null) {
        scaler = new MinMaxScaler().setInputCol("day_vec").setOutputCol("normalized_day")
        scalerModel = scaler.fit(assembledDayDf)
      }
      //returning scalar value instead of a vector
      val normalizedDayDf = scalerModel.transform(assembledDayDf).drop("day_vec")
      normalizedDayDf.withColumn("normalized_day", vectorToScalarUDF(col("normalized_day")))
    }


  //normalizing month, same principle as previous
  def normalizeMonth(dataFrame: DataFrame): DataFrame = {
    val assemblerMonth = new VectorAssembler().setInputCols(Array("month")).setOutputCol("month_vec")
    val assembledMonthDf = assemblerMonth.transform(dataFrame)
    if (scalerMonth == null) {
      scalerMonth = new MinMaxScaler().setInputCol("month_vec").setOutputCol("normalized_month")
      scalerModelMonth = scalerMonth.fit(assembledMonthDf)
    }
    val normalizedMonthDf = scalerModelMonth.transform(assembledMonthDf).drop("month_vec")
    normalizedMonthDf.withColumn("normalized_month", vectorToScalarUDF(col("normalized_month")))
  }

  //normalizing year, same principle as previous
  def normalizeYear(dataFrame: DataFrame): DataFrame = {
    val assemblerYear = new VectorAssembler().setInputCols(Array("year")).setOutputCol("year_vec")
    val assembledYearDf = assemblerYear.transform(dataFrame)
    if (scalerYear == null) {
      scalerYear = new MinMaxScaler().setInputCol("year_vec").setOutputCol("normalized_year")
      scalerModelYear = scalerYear.fit(assembledYearDf)
    }
    val normalizedYearDf = scalerModelYear.transform(assembledYearDf).drop("year_vec")
    normalizedYearDf.withColumn("normalized_year", vectorToScalarUDF(col("normalized_year")))
  }



  //day of the week will be transformed into 2 values, sin and cos of day of the week, to preserve relationship between for example sunday and monday
  def transformDayOfWeek(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("transformed_sin_day_of_week",
      sin(((col("day_of_week") - 1) * (2 * Math.PI / 7))).as("transformed_sin_day_of_week"))
      .withColumn("transformed_cos_day_of_week",
        cos(((col("day_of_week") - 1) * (2 * Math.PI / 7))).as("transformed_cos_day_of_week"))
  }


  //in the same way the hour will be transformed
  def transformedHour(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("transformed_sin_hour",
      sin(((col("hour") * 2 * Math.PI) / 24)).as("transformed_sin_hour"))
      .withColumn("transformed_cos_hour",
        cos(((col("hour") * 2 * Math.PI) / 24)).as("transformed_cos_hour"))
  }


  //transforming time slots in the same way
  def transformedTimeSlotMinutes(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("transformed_sin_time_slot",
      sin(((col("time_slot_minutes") * 2 * Math.PI) / 1440)).as("transformed_sin_time_slot"))
      .withColumn("transformed_cos_time_slot",
        cos(((col("time_slot_minutes") * 2 * Math.PI) / 1440)).as("transformed_cos_time_slot"))
  }

  //encoding pu locations
  def processPULocation(df: DataFrame): DataFrame = {
    //one hot encoding
    val encoder = new OneHotEncoder()
      .setInputCol("PULocationID")
      .setOutputCol("PULocationID_encoded")

    //creating transformation pipeline
    val pipeline = new Pipeline().setStages(Array(encoder))

    //fitting and transforming data
    pipelineModel = pipeline.fit(df)
    val dfTransformed = pipelineModel.transform(df)

    dfTransformed
  }


  //a method that calls all transformation data methods and drops all irelevant(replaced) columns
  def normalizeData(dataFrame: DataFrame): DataFrame = {
    val dayNormalizedDf = normalizeDay(dataFrame)
    val monthNormalizedDf = normalizeMonth(dayNormalizedDf)
    val yearNormalizedDf = normalizeYear(monthNormalizedDf)
    val hourWithSinCosDf = transformedHour(yearNormalizedDf)
    val timeSlotWithSinCosDf = transformedTimeSlotMinutes(hourWithSinCosDf)
    val encodeLocId = processPULocation(timeSlotWithSinCosDf)
    var finalDf = transformDayOfWeek(encodeLocId)
    finalDf = finalDf.drop("day", "month", "year", "hour", "time_slot_minutes", "day_of_week")

    finalDf
  }

  //sploting data into training 70%, test 15% and validation 15%
  //it takes in data frame data type but returns tuple of 3 data frames
  def splitData(dataFrame: DataFrame): (DataFrame, DataFrame, DataFrame) = {
    val Array(trainingData, validationData, testData) = dataFrame.randomSplit(Array(0.7, 0.15, 0.15), seed = 42)
    //same as before, to optimize computation
    trainingData.persist()
    validationData.persist()
    testData.persist()

    //printing their size
    println(s"Training data: ${trainingData.count()}")
    println(s"Validation data: ${validationData.count()}")
    println(s"Test data: ${testData.count()}")

    (trainingData, validationData, testData)
  }


  //assembling data (multiple columns into 1 vector column), it is what spark requires, it will be used the model
  def assembleFeatures(data: DataFrame): DataFrame = {
    //data that will be used as an input (all except for "demand")
    val xColumns = Array("PULocationID_encoded", "normalized_day", "normalized_month", "normalized_year",
      "transformed_sin_hour", "transformed_cos_hour",
      "transformed_sin_time_slot", "transformed_cos_time_slot",
      "red_date")

    //assembling data
    val assembler = new VectorAssembler()
      .setInputCols(xColumns)
      .setOutputCol("features")

    assembler.transform(data)
  }


  //preparing the data and training the model
  def trainModel(trainingData: DataFrame): LinearRegressionModel = {
    //assembling x-data(input data)
    val vectorTrainData = assembleFeatures(trainingData)

    //TRAINING THE MODEL using train data
    val lr = new LinearRegression()
      .setLabelCol("demand") //output
      .setFeaturesCol("features") //input
      .setRegParam(0.1) //reg. parameter to prevent overfitting
      .setElasticNetParam(0.3) //elasticnetparameter, regulation parameter

    //fitting the model
    lr.fit(vectorTrainData)
  }

  //evaluating model
  def evaluateModel(model: LinearRegressionModel, validationData: DataFrame, testData: DataFrame): Unit = {
    //evaluation for validation data
    //applying data on the model
    val validationDatafeatures = assembleFeatures(validationData)
    val predictions = model.transform(validationDatafeatures)
    //evaluation using RMSE
    val evaluator = new RegressionEvaluator()
      .setLabelCol("demand")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root mean squared error RMSE val. data = $rmse")

    //evaluation for test data, done in the same way
    val testDataWithFeatures = assembleFeatures(testData)
    val testPredictions = model.transform(testDataWithFeatures)
    val testRmse = evaluator.evaluate(testPredictions)
    println(s"Root mean squared error RMSE test data = $testRmse")
  }

  //main model for training and evaluation
  def linRegModel(trainingData: DataFrame, validationData: DataFrame, testData: DataFrame): LinearRegressionModel = {
    val model = trainModel(trainingData)
    evaluateModel(model, validationData, testData)
    model
  }

  //exploring data
  def inspectData(dataFrame: DataFrame): Unit = {
    println("Data schema:")
    dataFrame.printSchema()
    println("Summary statistics:")
    dataFrame.describe().show()
    println("Missing values:")
    dataFrame.select(dataFrame.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()
    println("First 5 rows:")
    dataFrame.show(5, false)
  }

  //for input data, features need to be scaled using same scalers as in the model
  //normalizing day
  //it also needs to create vector, scale data and then transform vector into scalar
  def normalizeInputDay(dataFrame: DataFrame): DataFrame={
    val df1 = new VectorAssembler().setInputCols(Array("day")).setOutputCol("day_vec")
    val df2 = df1.transform(dataFrame)
    val df3 = scalerModel.transform(df2).drop("day_vec")
    val df4 = df3.withColumn("normalized_day", vectorToScalarUDF(col("normalized_day")))
    df4
  }

  //normalizing month
  def normalizeInputMonth(dataFrame: DataFrame): DataFrame = {
    val df1 = new VectorAssembler().setInputCols(Array("month")).setOutputCol("month_vec")
    val df2 = df1.transform(dataFrame)
    val df3 = scalerModelMonth.transform(df2).drop("month_vec")
    val df4 = df3.withColumn("normalized_month", vectorToScalarUDF(col("normalized_month")))
    df4
  }

  //normalizing year
  def normalizeInputYear(dataFrame: DataFrame): DataFrame = {
    val df1 = new VectorAssembler().setInputCols(Array("year")).setOutputCol("year_vec")
    val df2 = df1.transform(dataFrame)
    val df3 = scalerModelYear.transform(df2).drop("year_vec")
    val df4 = df3.withColumn("normalized_year", vectorToScalarUDF(col("normalized_year")))
    df4
  }

  //on new data predictions will be done here
  def predictDemand(date: String, time: String, location: String, model: LinearRegressionModel): Unit = {
    //creating a df from input
    val df = spark.createDataFrame(Seq((date, time, location))).toDF("date", "time", "PULocationID") .withColumn("PULocationID", col("PULocationID").cast("bigint"))
    //combining date and time into 'tpep_pickup_datetime'
    val df1 = df.withColumn("tpep_pickup_datetime", to_timestamp(concat(col("date"), lit(" "), col("time")), "yyyy-MM-dd HH:mm:ss"))

    //using methods for preprocesing to pre-process data in thr same way
    val df2 = preprocessData(df1)
    val df3 = groupTimeSlots(df2)
    val df4 = encodeTimeSlots(df3)
    val df5 = addRedDays(df4)
    val df6 = normalizeInputDay(df5)
    val df7 = normalizeInputMonth(df6)
    val df8 = normalizeInputYear(df7)
    val df9 = transformedHour(df8)
    val df10 = transformedTimeSlotMinutes(df9)
    var finalInputDf = transformDayOfWeek(df10)
    val finalInputDf2 = pipelineModel.transform(finalInputDf) //encoding PU location

    finalInputDf2.show()

    //assembling features before training the model
    val assemledDf = assembleFeatures(finalInputDf2)

    //predicting demand
    val prediction = model.transform(assemledDf)
    //displaying demand
    prediction.select("prediction").show()
  }
}








