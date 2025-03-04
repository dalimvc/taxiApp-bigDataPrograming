import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object SparkApp {

  def main(args: Array[String]): Unit = {
    //
    val spark = SparkSession.builder
      .appName("MySparkApp")
      .master("local[*]")
      .getOrCreate()

    //setting command to reduce amount of information when running the code
    spark.sparkContext.setLogLevel("WARN")

    //paths with files that will be used
    val filePaths1 = (1 to 12).map(month => f"Data/yellow_tripdata_2024-$month%02d.parquet").toArray
    val filePaths2 = (1 to 12).map(month => f"Data/yellow_tripdata_2023-$month%02d.parquet").toArray
    val filePaths3 = (1 to 12).map(month => f"Data/yellow_tripdata_2022-$month%02d.parquet").toArray

    //leading data
    val df = loadData(spark, filePaths1 ++ filePaths2 ++ filePaths3)

    //pre-processing data
    val processedDf = df
      .transform(preprocessData)
      .transform(groupTimeSlots)
      .transform(addRedDays)

    //data exploration
    //inspectData(processedDf)

    spark.stop()
  }


  //leading the data sets, only 2 selected columns
  def loadData(spark: SparkSession, paths: Array[String]): DataFrame = {
    val dfs = paths.map { path =>
      spark.read
        .parquet(path)
        .select(
          col("tpep_pickup_datetime"),
          col("PULocationID").cast("bigint").alias("PULocationID")
        )
    }
    dfs.reduce(_ union _) //to help with different data types
  }


  // from tpep_pickup_datetime extracting date, time, and day of the week
  def preprocessData(df: DataFrame): DataFrame = {
    df.withColumn("date", to_date(col("tpep_pickup_datetime")))
      .withColumn("time", date_format(col("tpep_pickup_datetime"), "HH:mm:ss"))
      .withColumn("day_of_week", date_format(col("tpep_pickup_datetime"), "EEEE"))
      .drop("tpep_pickup_datetime")
  }


  //groping timeslots into 10 minutes intervals
  def groupTimeSlots(df: DataFrame): DataFrame = {
    val dfWithTimeSlots = df.withColumn("time_slot",
      expr("floor(minute(time) / 10) * 10") // rounding minutes to 10-minute intervals
    ).withColumn("time_slot",
      concat_ws(":", date_format(col("time"), "HH"), lpad(col("time_slot").cast("string"), 2, "0"))
    )
    // keeping "time_slot", "date", "day_of_week" columns
    dfWithTimeSlots.groupBy("PULocationID", "time_slot", "date", "day_of_week")
      .agg(count("*").alias("demand"))
      .orderBy("PULocationID", "time_slot", "date")
  }


  //adding red dates column with 1 it is a red date, 0 it is not
  def addRedDays(dataFrame: DataFrame): DataFrame = {
    //red days for 2024
    val redDays2024 = Set(
      "2024-01-01", "2024-01-15", "2024-02-12", "2024-02-19",
      "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
      "2024-10-14", "2024-11-05", "2024-11-11", "2024-11-28", "2024-12-25"
    )
    //red days for 2022
    val redDays2022 = Set(
      "2022-01-01", "2022-01-17", "2022-02-21", "2022-05-30",
      "2022-06-19", "2022-07-04", "2022-09-05", "2022-10-10",
      "2022-11-11", "2022-11-24", "2022-12-25"
    )
    //red days for 2023
    val redDays2023 = Set(
      "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29",
      "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09",
      "2023-11-11", "2023-11-23", "2023-12-25"
    )
    //combining red dates
    val allRedDays = redDays2024 ++ redDays2022 ++ redDays2023
    //checking if df contains red date
    val isRedDay = udf((date: String) => if (allRedDays.contains(date)) 1 else 0)
    //adding 0 for no match and 1 for a match
    dataFrame.withColumn("red_date", isRedDay(col("date")))
  }

  //data exploration
  def inspectData(dataFrame: DataFrame): Unit = {
    //printing nr. or rows
    println(s"Total rows: ${dataFrame.count()}")

    //printing details about data, data types
    println("Data schema:")
    dataFrame.printSchema()

    println("Summary statistics:")
    dataFrame.describe().show()

    println("Missing values:")
    dataFrame.select(dataFrame.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)):_*).show()

    println("First 5 rows:")
    dataFrame.show(5, truncate = false)
  }
}
