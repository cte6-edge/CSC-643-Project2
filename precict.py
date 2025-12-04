import sys
import zipfile
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

MODEL_ZIP_PATH = "/app/wine_model.zip"
MODEL_EXTRACT_DIR = "/tmp/wine_model"

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

ALL_COLUMNS = FEATURES + ["quality"]

# -----------------------------------------------------------------------------

def extract_model():
    if os.path.exists(MODEL_EXTRACT_DIR):
        shutil.rmtree(MODEL_EXTRACT_DIR)

    print(f"*** Extracting model to {MODEL_EXTRACT_DIR} ...")
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zf:
        zf.extractall("/tmp/")

    return MODEL_EXTRACT_DIR


def clean_csv(spark, path):
    print(f"*** Loading test data: {path}")

    raw = spark.read.csv(path, header=False, inferSchema=False)

    print("*** Raw columns:", raw.columns)

    # Drop header row (first row contains text)
    df = raw.filter(raw["_c0"] != '"""""fixed acidity""""')

    print(f"*** Total rows (including header row): {raw.count()}")
    print(f"*** Rows after dropping header: {df.count()}")

    # Keep only first 12 columns
    df = df.select([f"_c{i}" for i in range(12)])

    # Remove quotes, stray characters, and cast to float
    for i, name in enumerate(ALL_COLUMNS):
        df = df.withColumn(name, regexp_replace(col(f"_c{i}"), "[^0-9\\.]", ""))
        df = df.withColumn(name, when(col(name) == "", None).otherwise(col(name).cast("double")))

    print(f"*** Rows before dropping invalid numeric values: {df.count()}")
    df = df.dropna(subset=ALL_COLUMNS)
    print(f"*** Rows after dropping invalid numeric values:  {df.count()}")

    df = df.select(ALL_COLUMNS)

    print("\n*** Final schema used for prediction:")
    df.printSchema()

    print("\n*** Example cleaned rows:")
    df.show(5)

    return df


# -----------------------------------------------------------------------------

def compute_f1(df):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality",
        predictionCol="prediction",
        metricName="f1"
    )
    return evaluator.evaluate(df)


# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <TestDataset.csv>")
        sys.exit(1)

    test_path = sys.argv[1]
    print(f"*** Script arguments: {sys.argv}")

    spark = (
        SparkSession.builder.appName("CS643-Wine-Predict")
        .getOrCreate()
    )

    # Clean and load CSV
    df = clean_csv(spark, test_path)

    # Load model
    print(f"\n*** Loading model from zip: {MODEL_ZIP_PATH}")
    model_dir = extract_model()
    model = PipelineModel.load(model_dir)
    print("*** Model loaded.")

    # Run prediction
    print("*** Running predictions...\n")
    pred = model.transform(df)

    print("*** Sample predictions (first 10 rows):")
    pred.select(ALL_COLUMNS + ["prediction"]).show(10, truncate=False)

    # Compute F1 score
    print("*** Computing F1 score (good vs bad)...")
    f1 = compute_f1(pred)
    print(f"*** F1 score: {f1}")

    spark.stop()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
