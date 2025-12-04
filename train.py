import os
import zipfile
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

TRAIN_PATH = "/home/ubuntu/cs643/data/TrainingDataset.csv"
VALID_PATH = "/home/ubuntu/cs643/data/ValidationDataset.csv"
OUTPUT_DIR = "/home/ubuntu/cs643/artifacts"
MODEL_DIR = os.path.join(OUTPUT_DIR, "wine_model")
MODEL_ZIP = os.path.join(OUTPUT_DIR, "wine_model.zip")

# -----------------------------------------------------------------------------

def load_csv(spark, path):
    """Load CSV with headers using Spark."""
    df = (
        spark.read.csv(path, header=True, inferSchema=True)
        .withColumn("quality", col("quality").cast("double"))
    )
    return df


def zip_model_folder(model_dir, zip_path):
    """Zip the folder Spark creates for the model."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, os.path.dirname(model_dir))
                z.write(full, rel)


# -----------------------------------------------------------------------------

def main():
    spark = (
        SparkSession.builder.appName("CS643-Wine-Training")
        .getOrCreate()
    )

    print("\n*** Loading training data...")
    train_df = load_csv(spark, TRAIN_PATH)
    print(f"Training rows: {train_df.count()}")

    print("\n*** Loading validation data...")
    valid_df = load_csv(spark, VALID_PATH)
    print(f"Validation rows: {valid_df.count()}")

    FEATURES = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
        "density", "pH", "sulphates", "alcohol"
    ]

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")
    rf = RandomForestClassifier(featuresCol="features", labelCol="quality")

    print("\n*** Building pipeline and training model...")
    df_train_vec = assembler.transform(train_df)
    df_train_scaled = scaler.fit(df_train_vec).transform(df_train_vec)

    df_valid_vec = assembler.transform(valid_df)
    df_valid_scaled = scaler.fit(df_valid_vec).transform(df_valid_vec)

    model = rf.fit(df_train_scaled)

    print("\n*** Evaluating model on validation set...")
    preds = model.transform(df_valid_scaled)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator.evaluate(preds)
    print(f"\n*** F1 Score on Validation Set: {f1}")

    print(f"\n*** Saving model to {MODEL_DIR} ...")
    model.write().overwrite().save(MODEL_DIR)

    print(f"*** Zipping model to {MODEL_ZIP} ...")
    zip_model_folder(MODEL_DIR, MODEL_ZIP)

    print("\n*** Done. You can now SCP wine_model.zip to your docker-predict EC2.\n")

    spark.stop()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
