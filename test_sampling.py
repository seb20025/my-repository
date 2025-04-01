import pytest
from pyspark.sql import SparkSession
from test_sampling import get_sample  # Asegúrate de importar tu función correctamente

def test_get_sample():
    spark = SparkSession.builder.master("local[1]").appName("pytest-spark").getOrCreate()
    data = [(1, "A"), (2, "B"), (3, "C")]
    df = spark.createDataFrame(data, ["id", "name"])

    result = get_sample(df, 2)
    assert result.count() == 2
