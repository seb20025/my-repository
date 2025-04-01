# Importaciones de bibliotecas estándar de Python
import datetime
import logging
import os
import math
from datetime import datetime, timedelta
from functools import reduce
from itertools import combinations, combinations_with_replacement
import tempfile
import shap
import time
import random

# Bibliotecas de manejo de datos
import pandas as pd
import numpy as np
import geopandas as gpd
import missingno as msno

# Bibliotecas de visualización
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

# Bibliotecas de geolocalización
from geopy.geocoders import Nominatim
from shapely.geometry import Point
import requests

# Bibliotecas de Spark SQL y tipos de datos
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Bibliotecas de Spark ML
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import (
    RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, 
    LogisticRegression, NaiveBayes, LinearSVC, OneVsRest, 
    MultilayerPerceptronClassifier, FMClassifier
)
from xgboost.spark.estimator import SparkXGBClassifier, SparkXGBClassifierModel
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler, DenseVector, PCA
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.stat import Summarizer
from pyspark.ml.linalg import Vectors, VectorUDT, Vector
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline, Transformer, Estimator
from pyspark.ml.param.shared import Param, Params
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, SparkTrials
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import StorageLevel

# Bibliotecas de análisis de datos y modelado
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from scipy.stats import zscore, gaussian_kde
from sklearn.metrics import (
    roc_curve, roc_auc_score, mean_squared_error, 
    mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
)
from pyspark.sql.functions import col, sum as F_sum, expr

# Bibliotecas de MLflow
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

spark = SparkSession.builder \
    .appName("Pipeline") \
    .enableHiveSupport() \
    .getOrCreate()

#==================
#    Training
#==================
    
class DataSplit(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    """
    Transformer para dividir un DataFrame de Spark en conjuntos de entrenamiento, validación y prueba
    basado en una columna específica y valores predefinidos.

    :param column_split: Nombre de la columna utilizada para estratificar la división.
    :type column_split: str
    :param train_values: Lista de valores para el conjunto de entrenamiento.
    :type train_values: list
    :param test_value: Valor para el conjunto de prueba.
    :type test_value: str
    :param val_value: Valor para el conjunto de validación.
    :type val_value: str

    :raises ValueError: Si el DataFrame no contiene la columna de división especificada.
    :raises ValueError: Si no se especifican valores de entrenamiento, prueba o validación.
    """

    column_split = Param(Params._dummy(),"column_split","Nombre de la columna utilizada para estratificar la división.")
    train_values = Param(Params._dummy(),"train_values","Lista de valores para el conjunto de entrenamiento.")
    test_value = Param(Params._dummy(),"test_value","Valor para el conjunto de prueba.")
    val_value = Param(Params._dummy(),"val_value","Valor para el conjunto de validación.")

    def __init__(self, column_split: str = None, train_values: list = None, test_value: str = None, val_value: str = None):
        """
        Inicializa el Transformer con los parámetros necesarios para dividir el DataFrame.

        :param column_split: Nombre de la columna utilizada para estratificar la división.
        :type column_split: str, optional
        :param train_values: Lista de valores para el conjunto de entrenamiento.
        :type train_values: list, optional
        :param test_value: Valor para el conjunto de prueba.
        :type test_value: str, optional
        :param val_value: Valor para el conjunto de validación.
        :type val_value: str, optional
        """
        super(DataSplit, self).__init__()
        self._setDefault(column_split=None, train_values=[], test_value=None, val_value=None)
        if column_split:
            self._set(column_split=column_split)
        if train_values:
            self._set(train_values=train_values)
        if test_value:
            self._set(test_value=test_value)
        if val_value:
            self._set(val_value=val_value)

    def _transform(self, dataset: DataFrame) -> tuple:
        """
        Divide el DataFrame en conjuntos de entrenamiento, validación y prueba según los valores configurados.

        :param dataset: DataFrame de Spark que deseas dividir.
        :type dataset: pyspark.sql.DataFrame

        :return: Tuple con DataFrames de Spark para entrenamiento, prueba y validación.
        :rtype: tuple (train_df, test_df, val_df)

        :raises ValueError: Si el DataFrame no contiene la columna de división especificada.
        :raises ValueError: Si no se especifican valores de entrenamiento, prueba o validación.
        """
        column_split = self.getOrDefault(self.column_split)
        train_values = self.getOrDefault(self.train_values)
        test_value = self.getOrDefault(self.test_value)
        val_value = self.getOrDefault(self.val_value)

        # Verificar que la columna de división existe en el DataFrame
        if column_split not in dataset.columns:
            raise ValueError(f"[ERROR] The specified column '{column_split}' is not present in the DataFrame.")

        # Verificar que se han especificado valores de división
        if not train_values or test_value is None or val_value is None:
            raise ValueError("[ERROR] You must specify at least one value for training, testing, and validation.")

        # Filtrar los conjuntos de datos
        train_df = dataset.filter(F.col(column_split).isin(train_values))
        test_df = dataset.filter(F.col(column_split) == test_value)
        val_df = dataset.filter(F.col(column_split) == val_value)

        return train_df, test_df, val_df
    
    @staticmethod
    def _transform(df: DataFrame, column_split: str, train_values: list, test_value: str, val_value: str) -> tuple:
        """
        Divide el DataFrame en conjuntos de entrenamiento, validación y prueba según los valores configurados.

        :param dataset: DataFrame de Spark que deseas dividir.
        :return: Tuple con DataFrames de Spark para entrenamiento, prueba y validación.
        """
        train_df = df.filter(F.col(column_split).isin(train_values))
        test_df = df.filter(F.col(column_split) == test_value)
        val_df = df.filter(F.col(column_split) == val_value)

        return train_df, test_df, val_df
    
class AuxRenamePredictionColumn(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer para renombrar columnas en un DataFrame de Spark.

    :param columns_to_rename: Diccionario con las columnas a renombrar en formato {columna_original: columna_nueva}.
    :type columns_to_rename: dict

    :raises ValueError: Si el diccionario de columnas a renombrar no está definido o está vacío.
    :raises ValueError: Si todas las columnas especificadas en `columns_to_rename` no están en el DataFrame.
    """

    columns_to_rename = Param(Params._dummy(),"columns_to_rename","Diccionario de columnas a renombrar en el formato {columna_original: columna_nueva}.")

    def __init__(self, columns_to_rename: dict = None):
        """
        Inicializa el Transformer con el diccionario de columnas a renombrar.

        :param columns_to_rename: Diccionario {columna_original: columna_nueva}.
        :type columns_to_rename: dict, optional
        """
        super(AuxRenamePredictionColumn, self).__init__()
        self._setDefault(columns_to_rename={})
        if columns_to_rename:
            self._set(columns_to_rename=columns_to_rename)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Renombra las columnas en un DataFrame según el diccionario especificado.

        :param dataset: DataFrame de Spark donde se aplicará el renombramiento.
        :type dataset: pyspark.sql.DataFrame

        :return: DataFrame con las columnas renombradas.
        :rtype: pyspark.sql.DataFrame

        :raises ValueError: Si el diccionario de columnas a renombrar no está definido o está vacío.
        :raises ValueError: Si todas las columnas especificadas en `columns_to_rename` no están en el DataFrame.
        """
        columns_to_rename = self.getOrDefault(self.columns_to_rename)

        if not columns_to_rename:
            raise ValueError("[ERROR] The dictionary `columns_to_rename` is empty. Please provide valid column names.")

        missing_columns = [col for col in columns_to_rename.keys() if col not in dataset.columns]

        if len(missing_columns) == len(columns_to_rename):
            raise ValueError(f"[ERROR] None of the specified columns {list(columns_to_rename.keys())} were found in the DataFrame.")

        for original_col, new_col in columns_to_rename.items():
            if original_col in dataset.columns:
                dataset = dataset.withColumnRenamed(original_col, new_col)
            else:
                print(f"[WARNING] The column '{original_col}' was not found in the DataFrame and was not renamed.")

        return dataset.persist(StorageLevel.MEMORY_AND_DISK)
    
class AuxCombinePredictions(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer para combinar predicciones de múltiples modelos en un solo resultado consolidado.

    :param raw_prediction_cols: Lista de columnas `rawPrediction` para combinar.
    :type raw_prediction_cols: list
    :param probability_cols: Lista de columnas `probability` para combinar.
    :type probability_cols: list
    :param threshold: Umbral de probabilidad para la clasificación final.
    :type threshold: float, optional

    :raises ValueError: Si ninguna de las columnas `rawPrediction` o `probability` está disponible en el DataFrame.
    :raises ValueError: Si no se pueden combinar las columnas especificadas.
    """

    raw_prediction_cols = Param(Params._dummy(),"raw_prediction_cols","Lista de columnas de rawPrediction para combinar.")
    probability_cols = Param(Params._dummy(),"probability_cols","Lista de columnas de probability para combinar.")
    threshold = Param(Params._dummy(),"threshold","Umbral de probabilidad para clasificación.")

    def __init__(self, raw_prediction_cols=None, probability_cols=None, threshold=0.5):
        """
        Inicializa el Transformer con las listas de columnas a combinar y el umbral de clasificación.

        :param raw_prediction_cols: Lista de columnas `rawPrediction` a combinar.
        :type raw_prediction_cols: list, optional
        :param probability_cols: Lista de columnas `probability` a combinar.
        :type probability_cols: list, optional
        :param threshold: Umbral de clasificación basado en `probability`.
        :type threshold: float, optional
        """
        super(AuxCombinePredictions, self).__init__()
        self._setDefault(raw_prediction_cols=[], probability_cols=[], threshold=0.5)
        self._set(
            raw_prediction_cols=raw_prediction_cols or [],
            probability_cols=probability_cols or [],
            threshold=threshold
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Aplica la combinación de columnas de predicción y probabilidad.

        :param dataset: DataFrame con las predicciones individuales de múltiples modelos.
        :type dataset: pyspark.sql.DataFrame

        :return: DataFrame con las predicciones combinadas.
        :rtype: pyspark.sql.DataFrame

        :raises ValueError: Si ninguna de las columnas `rawPrediction` o `probability` está disponible en el DataFrame.
        :raises ValueError: Si no se pueden combinar las columnas especificadas.
        """
        # Obtener parámetros
        raw_prediction_cols = self.getOrDefault(self.raw_prediction_cols)
        probability_cols = self.getOrDefault(self.probability_cols)
        threshold = self.getOrDefault(self.threshold)

        # Función para combinar rawPrediction
        def combine_raw(*arrays):
            return [sum(x) for x in zip(*arrays)]

        # Función para combinar probability
        def combine_prob(*arrays):
            num_models = len(arrays)
            return [sum(x) / num_models for x in zip(*arrays)]

        # Función para convertir array<double> a Vector
        def to_vector(array):
            return Vectors.dense(array)

        # UDFs para combinación y conversión
        combine_raw_udf = F.udf(combine_raw, ArrayType(DoubleType()))
        combine_prob_udf = F.udf(combine_prob, ArrayType(DoubleType()))
        to_vector_udf = F.udf(to_vector, VectorUDT())

        # Procesar rawPrediction
        valid_raw_cols = [col for col in raw_prediction_cols if col in dataset.columns]
        if not valid_raw_cols:
            print(f"Expected rawPrediction columns: {self.getOrDefault(self.raw_prediction_cols)}")
            print(f"Available columns: {dataset.columns}")
            raise ValueError("No valid rawPrediction columns found to combine.")
        if valid_raw_cols:
            dataset = dataset.withColumn(
                "rawPrediction_array",
                combine_raw_udf(*[F.col(c) for c in valid_raw_cols])
            ).withColumn(
                "rawPrediction",
                to_vector_udf(F.col("rawPrediction_array"))
            ).drop("rawPrediction_array")
        else:
            raise ValueError("No valid rawPrediction columns found to combine.")

        # Procesar probability
        valid_prob_cols = [col for col in probability_cols if col in dataset.columns]
        if not valid_prob_cols:
            print(f"Expected rawPrediction columns: {self.getOrDefault(self.probability_cols)}")
            print(f"Available columns: {dataset.columns}")
            raise ValueError("No valid rawPrediction columns found to combine.")
        if valid_prob_cols:
            num_models = len(valid_prob_cols)
            dataset = dataset.withColumn(
                "probability_array",
                combine_prob_udf(*[F.col(c) for c in valid_prob_cols])
            ).withColumn(
                "probability",
                to_vector_udf(F.col("probability_array"))
            ).drop("probability_array"
            ).withColumn(
                "prediction",
                F.when(vector_to_array(F.col("probability"))[1] >= 0.5, 1).otherwise(0)
            )
        else:
            raise ValueError("No valid probability columns found to combine.")

        return dataset
    
class TrainClassificatorModel(Estimator):
    """
    Estimator para entrenar modelos de clasificación y seleccionar el mejor o un conjunto de los mejores modelos.

    :param target_col: Columna objetivo en el DataFrame.
    :type target_col: str
    :param features_col: Columna de características en el DataFrame.
    :type features_col: str
    :param experiment_name: Nombre del experimento en MLflow.
    :type experiment_name: str
    :param catalog: Nombre del catálogo donde se almacenará el modelo.
    :type catalog: str
    :param schema: Nombre del esquema dentro del catálogo.
    :type schema: str
    :param model_name_publish: Nombre del modelo publicado en MLflow.
    :type model_name_publish: str
    :param thresholds: Lista de umbrales de probabilidad para clasificación.
    :type thresholds: list
    :param metric_name: Nombre de la métrica de evaluación (ej. "areaUnderROC", "f1", "precision").
    :type metric_name: str
    :param model_list: Lista de modelos a evaluar.
    :type model_list: list
    :param time_limit: Tiempo máximo de entrenamiento en formato 'HH:MM:SS' o 'MM:SS'.
    :type time_limit: str

    :raises ValueError: Si los parámetros no están bien definidos o si falta algún modelo.
    :raises Exception: Si ocurre un error inesperado durante el entrenamiento.
    """
    target_col = Param(Params._dummy(), "target_col", "Columna objetivo.")
    features_col = Param(Params._dummy(), "features_col", "Columna de características.")
    experiment_name = Param(Params._dummy(), "experiment_name", "Nombre del experimento en MLflow.")
    catalog = Param(Params._dummy(), "catalog", "Nombre del catalogo.")
    schema = Param(Params._dummy(), "schema", "Nombre del esquema.")
    model_name_publish = Param(Params._dummy(), "model_name_publish", "Nombre del modelo.")
    thresholds = Param(Params._dummy(), "thresholds", "Lista de umbrales de probabilidad para clasificación.")
    metric_name = Param(Params._dummy(), "metric_name", "Nombre de la métrica de evaluación.")
    model_list = Param(Params._dummy(), "model_list", "Lista de modelos a seleccionar en UniqueModel.")
    time_limit = Param(Params._dummy(), "time_limit", "Tiempo máximo de entrenamiento en formato 'HH:MM:SS', 'MM:SS'.")

    def __init__(self, target_col: str, features_col: str, experiment_name: str, catalog: str, schema: str, 
                 model_name_publish: str, thresholds: list, metric_name: str, model_list: list, time_limit: str):
        """
        Inicializa la clase `TrainClassificatorModel`.

        :param target_col: Nombre de la columna objetivo.
        :param features_col: Nombre de la columna de características.
        :param experiment_name: Nombre del experimento en MLflow.
        :param catalog: Nombre del catálogo donde se almacenará el modelo.
        :param schema: Nombre del esquema dentro del catálogo.
        :param model_name_publish: Nombre del modelo publicado en MLflow.
        :param thresholds: Lista de umbrales para clasificación.
        :param metric_name: Métrica de evaluación a optimizar.
        :param model_list: Lista de modelos a entrenar.
        :param time_limit: Tiempo máximo de entrenamiento en formato 'HH:MM:SS' o 'MM:SS'.
        """
        super(TrainClassificatorModel, self).__init__()
        self._set(target_col=target_col, features_col=features_col, experiment_name=experiment_name, 
                  catalog=catalog, schema=schema, model_name_publish=model_name_publish, thresholds=thresholds, 
                  metric_name=metric_name, model_list=model_list, time_limit=time_limit)

    def _fit(self, dataset: tuple):
        """
        Entrena múltiples modelos de clasificación y selecciona el mejor modelo.

        :param dataset: Tupla con (train_df, test_df).
        :type dataset: tuple (pyspark.sql.DataFrame, pyspark.sql.DataFrame)

        :return: Resultados del mejor modelo seleccionado.
        :rtype: dict

        :raises ValueError: Si los parámetros son incorrectos o el DataFrame no tiene los datos necesarios.
        :raises Exception: Si ocurre un error inesperado durante el entrenamiento.
        """
        train_df, test_df = dataset
        target_col = self.getOrDefault(self.target_col)
        features_col = self.getOrDefault(self.features_col)
        experiment_name = self.getOrDefault(self.experiment_name)
        catalog = self.getOrDefault(self.catalog)
        schema = self.getOrDefault(self.schema)
        model_name_publish = self.getOrDefault(self.model_name_publish)
        thresholds = self.getOrDefault(self.thresholds)
        metric_name = self.getOrDefault(self.metric_name)
        model_list = [model.upper() for model in self.getOrDefault(self.model_list)]
        time_limit = self.getOrDefault(self.time_limit)

        try:
            # Especifica la ruta del archivo de log
            log_directory = '../logs'
            os.makedirs(log_directory, exist_ok=True)
            log_file_path = os.path.join(log_directory, 'model_training.log')

            # Configura el logging solo si el manejador aún no existe
            logger = logging.getLogger()
            if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_path for handler in logger.handlers):
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(logging.ERROR)
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(file_handler)
                logger.setLevel(logging.ERROR)

            # Validar tamaño del vector de características
            feature_vector_size = train_df.select(features_col).first()[0].size
            print(f"[INFO]Feature vector size: {feature_vector_size}")

            # Validar número de clases
            num_classes = train_df.select(target_col).distinct().count()
            print(f"[INFO]Number of Classes: {num_classes}")

            @staticmethod
            def normalize_raw_prediction(raw_prediction: Vector) -> list[float]:
                """
                Normaliza rawPrediction para obtener probabilidades.

                :param raw_prediction: Vector con los valores de `rawPrediction`.
                :type raw_prediction: pyspark.ml.linalg.Vector
                :return: Lista de valores normalizados como probabilidades.
                :rtype: list[float]

                :raises ValueError: Si `raw_prediction` no es un vector válido.
                """
                if raw_prediction is None:
                    raise ValueError("[ERROR] The input `rawPrediction` cannot be None.")

                try:
                    array = raw_prediction.toArray()
                    total = sum(array)
                    return [x / total for x in array] if total > 0 else array
                except Exception as e:
                    raise ValueError(f"[ERROR] Failed to normalize `rawPrediction`: {e}")

            @staticmethod
            def add_probability_column(predictions: DataFrame) -> DataFrame:
                """
                Agrega la columna `probability` al DataFrame si no existe.

                :param predictions: DataFrame con predicciones del modelo.
                :type predictions: pyspark.sql.DataFrame
                :return: DataFrame con la columna `probability` agregada si no existía.
                :rtype: pyspark.sql.DataFrame

                :raises ValueError: Si el DataFrame no contiene `rawPrediction` y `probability`.
                """
                if predictions is None:
                    raise ValueError("[ERROR] The input `predictions` cannot be None.")

                if 'probability' not in predictions.columns:
                    if 'rawPrediction' in predictions.columns:
                        # Normalizar `rawPrediction` para crear `probability`
                        normalize_raw_prediction_udf = F.udf(
                            lambda raw: normalize_raw_prediction(raw), ArrayType(DoubleType())
                        )

                        predictions = predictions.withColumn(
                            'probability',
                            normalize_raw_prediction_udf(F.col('rawPrediction'))
                        )
                    else:
                        raise ValueError("[ERROR] The model does not provide either 'probability' or 'rawPrediction'. Probabilities cannot be computed.")

                return predictions
            
            @staticmethod
            def generate_model_key(model_name: str, params: dict) -> str:
                """
                Genera una clave única para identificar cada modelo en los trials.

                :param model_name: Nombre del modelo.
                :type model_name: str
                :param params: Diccionario de hiperparámetros del modelo.
                :type params: dict
                :return: Clave única en formato `model_name_param1=value1_param2=value2_...`.
                :rtype: str

                :raises ValueError: Si `model_name` está vacío o no es una cadena.
                :raises ValueError: Si `params` no es un diccionario.
                """
                if not isinstance(model_name, str) or not model_name.strip():
                    raise ValueError("[ERROR] `model_name` must be a non-empty string.")

                if not isinstance(params, dict):
                    raise ValueError("[ERROR] `params` must be a dictionary.")

                params_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
                return f"{model_name}_{params_str}"

            @staticmethod
            def generate_top_models(trials: list, model_list: list, top_n: int = 20) -> list:
                """
                Selecciona los top N modelos con la menor pérdida (`loss`) para cada modelo en `model_list`.

                :param trials: Lista de resultados de entrenamiento, donde cada elemento es un diccionario con información del modelo.
                :type trials: list[dict]
                :param model_list: Lista de nombres de modelos a evaluar.
                :type model_list: list[str]
                :param top_n: Número máximo de modelos a seleccionar por cada tipo de modelo.
                :type top_n: int, optional (default = 20)
                
                :return: Lista de los mejores modelos seleccionados.
                :rtype: list[dict]

                :raises ValueError: Si `trials` no es una lista.
                :raises ValueError: Si `model_list` no es una lista de cadenas.
                :raises ValueError: Si `top_n` no es un entero positivo.
                """
                if not isinstance(trials, list):
                    raise ValueError("[ERROR] `trials` must be a list of dictionaries containing model results.")

                if not isinstance(model_list, list) or not all(isinstance(model, str) for model in model_list):
                    raise ValueError("[ERROR] `model_list` must be a list of model names (strings).")

                if not isinstance(top_n, int) or top_n <= 0:
                    raise ValueError("[ERROR] `top_n` must be a positive integer.")

                top_models = []
                for model_name in model_list:
                    # Filtrar resultados por nombre de modelo
                    model_trials = [trial['result'] for trial in trials if trial['result']['model_name'] == model_name]

                    # Ordenar por pérdida y seleccionar los mejores N
                    sorted_models = sorted(model_trials, key=lambda x: x['loss'])[:top_n]

                    for trial_result in sorted_models:
                        key = generate_model_key(model_name, trial_result['params'])
                        top_models.append({
                            'key': key,
                            'name': model_name,
                            'params': trial_result['params'],
                            'model': trial_result['model'],
                            'loss': trial_result['loss']
                        })

                return top_models
            
            @staticmethod
            def generate_combinations_with_params(top_models: list, n_range: tuple = (2, 5)) -> list:
                """
                Genera combinaciones aleatorias de modelos a partir de los mejores modelos seleccionados (`top_models`).
                La selección de combinaciones se realiza con un shuffle equitativo basado en el tamaño de las combinaciones.

                :param top_models: Lista de los mejores modelos seleccionados.
                :type top_models: list[dict]
                :param n_range: Rango de tamaños de combinaciones a generar (mínimo, máximo).
                :type n_range: tuple[int, int], optional (default = (2, 5))

                :return: Lista de combinaciones de modelos seleccionadas aleatoriamente.
                :rtype: list[list[dict]]

                :raises ValueError: Si `top_models` no es una lista o está vacía.
                :raises ValueError: Si `n_range` no es una tupla de dos enteros positivos.
                :raises ValueError: Si el rango `n_range` es inválido (mínimo mayor que el máximo).
                """
                if not isinstance(top_models, list) or not top_models:
                    raise ValueError("[ERROR] `top_models` must be a non-empty list of models.")

                if not isinstance(n_range, tuple) or len(n_range) != 2 or not all(isinstance(n, int) and n > 0 for n in n_range):
                    raise ValueError("[ERROR] `n_range` must be a tuple of two positive integers (min, max).")

                if n_range[0] > n_range[1]:
                    raise ValueError("[ERROR] Invalid `n_range`: minimum value cannot be greater than maximum.")

                combinations_list = []
                weight_list = []

                # Generar combinaciones y asignar pesos inversos al número total de combinaciones por tamaño
                for n in range(n_range[0], n_range[1] + 1):
                    combinations = list(combinations_with_replacement(top_models, n))
                    combinations_list.extend(combinations)
                    # Peso inversamente proporcional al número de combinaciones posibles para este tamaño
                    weight_list.extend([1 / len(combinations)] * len(combinations))

                # Mezclar las combinaciones usando las probabilidades ajustadas
                combined = list(zip(combinations_list, weight_list))
                random.shuffle(combined)

                # Normalizar pesos para usar en una selección ponderada
                total_weight = sum(weight for _, weight in combined)
                weights = [weight / total_weight for _, weight in combined]

                # Seleccionar aleatoriamente respetando los pesos
                shuffled_combinations = random.choices(
                    [combo for combo, _ in combined],
                    weights=weights,
                    k=len(combinations_list)
                )

                return shuffled_combinations
            
            @staticmethod
            def _convert_time_to_seconds(time_str: str) -> int:
                """
                Convierte una cadena de tiempo en formato `'HH:MM:SS'` o `'MM:SS'` a segundos.

                :param time_str: Tiempo en formato `'HH:MM:SS'` o `'MM:SS'`.
                :type time_str: str
                :return: Tiempo en segundos.
                :rtype: int

                :raises ValueError: Si `time_str` no es una cadena válida.
                :raises ValueError: Si el formato no es `'HH:MM:SS'` o `'MM:SS'`.
                """
                if not isinstance(time_str, str):
                    raise ValueError("[ERROR] `time_str` must be a string.")

                parts = time_str.split(":")
                
                try:
                    parts = list(map(int, parts))
                except ValueError:
                    raise ValueError("[ERROR] `time_str` must contain only numeric values separated by colons.")

                if len(parts) == 3:
                    return int(timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2]).total_seconds())
                elif len(parts) == 2:
                    return int(timedelta(minutes=parts[0], seconds=parts[1]).total_seconds())
                else:
                    raise ValueError("[ERROR] Invalid time format. Use 'HH:MM:SS' or 'MM:SS'.")
            
            start_time = time.time()
            # Extract model name
            if time_limit >= 3600:
                rounded_time = round(time_limit / 3600)
                unit = "hour" if rounded_time == 1 else "hours"
            elif time_limit >= 60:
                rounded_time = round(time_limit / 60)
                unit = "minute" if rounded_time == 1 else "minutes"

            # Imprimir el mensaje formateado
            print(f"[INFO] Starting {rounded_time} {unit} training")

            @staticmethod
            def objective(params: dict) -> str:
                """
                Objective function for Hyperopt: trains a model and returns its metric.
                """
                elapsed_time = time.time() - start_time
                if elapsed_time > (time_limit/2):
                    print("[INFO] Ending the search for the time limit reached.")
                    raise StopIteration("[INFO] Time limit reached.")

                model_name = params.pop('model', '').upper()

                if model_name not in model_list:
                    raise ValueError(f"Model '{model_name}' is not allowed. Allowed models: {model_list}")

                # Determinar el mejor umbral para cada métrica
                def find_best_threshold(metrics, metric_name, maximize=True):
                    filtered_metrics = [m for m in metrics if m[metric_name] is not None]
                    if not filtered_metrics:
                        return None
                    best_metric = max(filtered_metrics, key=lambda x: x[metric_name]) if maximize else min(filtered_metrics, key=lambda x: x[metric_name])
                    return best_metric["threshold"]

                # Encontrar y registrar los mejores umbrales
                best_threshold_accuracy = find_best_threshold(threshold_metrics, "accuracy", maximize=True)
                best_threshold_precision = find_best_threshold(threshold_metrics, "precision", maximize=True)
                best_threshold_recall = find_best_threshold(threshold_metrics, "recall", maximize=True)
                best_threshold_f1 = find_best_threshold(threshold_metrics, "f1_score", maximize=True)

                if best_threshold_accuracy is not None:
                    mlflow.log_metric("best_threshold_accuracy", best_threshold_accuracy)
                if best_threshold_precision is not None:
                    mlflow.log_metric("best_threshold_precision", best_threshold_precision)
                if best_threshold_recall is not None:
                    mlflow.log_metric("best_threshold_recall", best_threshold_recall)
                if best_threshold_f1 is not None:
                    mlflow.log_metric("best_threshold_f1_score", best_threshold_f1)

                # Grafica de los diferentes umbrales de la predicción
                thresholds_plt = [item["threshold"] for item in threshold_metrics]
                accuracy_plt = [item["accuracy"] for item in threshold_metrics]
                precision_plt = [item["precision"] for item in threshold_metrics]
                recall_plt = [item["recall"] for item in threshold_metrics]
                f1_score_plt = [item["f1_score"] for item in threshold_metrics]

                plt.figure(figsize=(10, 6))
                plt.plot(thresholds_plt, accuracy_plt, label="Accuracy", marker="o")
                plt.plot(thresholds_plt, precision_plt, label="Precision", marker="o")
                plt.plot(thresholds_plt, recall_plt, label="Recall", marker="o")
                plt.plot(thresholds_plt, f1_score_plt, label="F1 Score", marker="o")

                plt.title("Variación de Métricas según el Threshold", fontsize=14)
                plt.xlabel("Threshold", fontsize=12)
                plt.ylabel("Valor de la Métrica", fontsize=12)
                plt.legend()
                plt.grid(True)

                # Guardar gráfica en un archivo temporal y registrarla en MLflow
                temp_file_path = os.path.join(tempfile.gettempdir(), 'Threshold.png')
                plt.savefig(temp_file_path)
                plt.close()
                mlflow.log_artifact(temp_file_path)
                os.remove(temp_file_path)

                # Guardar métricas en el artefacto de MLflow
                threshold_metrics_df = pd.DataFrame(threshold_metrics)
                threshold_metrics_path = os.path.join(tempfile.gettempdir(), "threshold_metrics.csv")
                threshold_metrics_df.to_csv(threshold_metrics_path, index=False)
                mlflow.log_artifact(threshold_metrics_path, artifact_path="threshold_analysis")
                os.remove(threshold_metrics_path)

                mlflow.log_metric("accuracy", accuracy if accuracy is not None else 0.0)
                mlflow.log_metric("precision", precision if precision is not None else 0.0)
                mlflow.log_metric("recall", recall if recall is not None else 0.0)
                mlflow.log_metric("f1_score", f1 if f1 is not None else 0.0)
                mlflow.log_metric("roc_auc", roc_auc if roc_auc is not None else 0.0)

                # Generar gráfica ROC
                fpr, tpr, _ = roc_curve(y_true, probas)
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")

                # Guardar gráfica en un archivo temporal y registrarla en MLflow
                temp_file_path = os.path.join(tempfile.gettempdir(), 'roc_curve.png')
                plt.savefig(temp_file_path)
                plt.close()
                mlflow.log_artifact(temp_file_path)
                os.remove(temp_file_path)

                # Generar y guardar matriz de confusión
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(8, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Clase 0", "Clase 1"],
                            yticklabels=["Clase 0", "Clase 1"])
                plt.xlabel("Predicción")
                plt.ylabel("Real")
                plt.title("Matriz de Confusión")

                # Guardar gráfica como artefacto
                temp_file_path = os.path.join(tempfile.gettempdir(), "confusion_matrix.png")
                plt.savefig(temp_file_path)
                plt.close()
                mlflow.log_artifact(temp_file_path, artifact_path="confusion_matrix")
                os.remove(temp_file_path)

                results = {
                    "best_model_name": best_model,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                    "lift": lift
                }

            return results

        except Exception as e:
            logging.error('Error al entrenar el modelo de clasificación', exc_info=True)
            return {"error": str(e)}
        
        
class ClassificatorValidator(Transformer, Params):
    """
    Transformer para validar un modelo de clasificación en Spark. Carga un modelo desde MLflow,
    genera predicciones sobre un DataFrame dado y calcula métricas de evaluación.
    """

    model_uri = Param(Params._dummy(), "model_uri", "URI del modelo registrado en MLflow.")
    id_col = Param(Params._dummy(), "id_col", "Columna de identificación de la cuenta.")
    prediction_col = Param(Params._dummy(), "prediction_col", "Nombre de la columna de predicción.")
    metric_reference = Param(
        Params._dummy(), "metric_reference", "Métrica de referencia a optimizar."
    )
    model_experiment = Param(Params._dummy(), "model_experiment", "Experiment donde se registró el modelo.")
    model_version = Param(Params._dummy(), "model_version", "Versión del modelo a cargar.")
    label_col = Param(Params._dummy(), "label_col", "Nombre de la columna de etiquetas reales.")

    def __init__(
        self,
        model_uri: str,
        id_col: str,
        prediction_col: str,
        metric_reference: str,
        model_experiment: str,
        model_version: int,
        label_col: str,
    ):
        super(ClassificatorValidator, self).__init__()
        self._set(
            model_uri=model_uri,
            id_col=id_col,
            prediction_col=prediction_col,
            metric_reference=metric_reference,
            model_experiment=model_experiment,
            model_version=model_version,
            label_col=label_col,
        )

    def _transform(self, dataset: DataFrame) -> dict:
        """
        Genera predicciones utilizando el modelo clasificador cargado desde MLflow y calcula métricas de evaluación.
        """
        try:
            model_uri = self.getOrDefault(self.model_uri)
            model_version = self.getOrDefault(self.model_version)
            prediction_col = self.getOrDefault(self.prediction_col)
            id_col = self.getOrDefault(self.id_col)
            metric_reference = self.getOrDefault(self.metric_reference)
            label_col = self.getOrDefault(self.label_col)
            model_experiment = self.getOrDefault(self.model_experiment)

            # Cargar el modelo desde MLflow
            try:
                mlflow.set_registry_uri('databricks-uc')
                mlflow.get_tracking_uri()
                loaded_model = mlflow.spark.load_model(f"{model_uri}_v{model_version}")
            except Exception as e:
                raise ValueError(f"[ERROR] Unable to load model from MLflow: {e}")

            # Generar predicciones
            predicts = loaded_model.transform(dataset)

            experiments = mlflow.search_experiments()
            experiment_id = None

            for exp in experiments:
                if exp.name == model_experiment:
                    experiment_id = exp.experiment_id
                    break

            # Verificar si el experimento existe
            if experiment_id is None:
                print(f"[INFO] Experiment '{os.path.basename(model_experiment)}' does not exist.")
                
            else:
                # Obtener todos los runs asociados a ese experimento y ordenarlos por 'start_time' (más reciente primero)
                runs = mlflow.search_runs(experiment_ids=[experiment_id])
                runs = runs.sort_values(by="start_time", ascending=False)

                # Iterar sobre los runs
                for _, row in runs.iterrows():
                    run_id = row["run_id"]
                    client = mlflow.MlflowClient()
                    execution = client.get_run(run_id)
                    metrics = execution.data.metrics
                    best_threshold = metrics.get(f"best_threshold_{metric_reference}")
                    
                    if best_threshold is not None:
                        break
                    
                if best_threshold is None:
                    raise ValueError("[ERROR] Best threshold not found in MLflow metrics.")

            # Convertir el vector de probabilidad a un array y calcular predicciones basadas en el umbral
            predicts = predicts.withColumn("prob_array", vector_to_array("probability"))
            predicts = predicts.withColumn("probabilidad_clase_1", F.col("prob_array")[1])
            predicts = predicts.withColumn(
                prediction_col,
                F.when(F.col("probabilidad_clase_1") >= best_threshold, 1).otherwise(0),
            )

            if id_col is not None:
                predicts = predicts.select(
                    id_col, prediction_col, "probabilidad_clase_1", label_col
                )
            else:
                predicts = predicts.select(prediction_col, "probabilidad_clase_1", label_col)

            # Evaluar métricas
            evaluator = BinaryClassificationEvaluator(
                labelCol=label_col,
                rawPredictionCol="probabilidad_clase_1",
                metricName="areaUnderROC",
            )
            roc_auc = evaluator.evaluate(predicts)

            evaluator_pr = BinaryClassificationEvaluator(
                labelCol=label_col,
                rawPredictionCol="probabilidad_clase_1",
                metricName="areaUnderPR",
            )
            pr_auc = evaluator_pr.evaluate(predicts)

            # Cálculo mejorado de métricas
            true_positives = predicts.filter(
                (F.col(prediction_col) == 1) & (F.col(label_col) == 1)
            ).count()
            false_positives = predicts.filter(
                (F.col(prediction_col) == 1) & (F.col(label_col) == 0)
            ).count()
            false_negatives = predicts.filter(
                (F.col(prediction_col) == 0) & (F.col(label_col) == 1)
            ).count()
            total_count = predicts.count()

            accuracy = (
                predicts.filter(F.col(prediction_col) == F.col(label_col)).count() / total_count
                if total_count > 0
                else 0
            )
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Obtener valores reales y predichos
            y_true = np.array(predicts.select(label_col).rdd.flatMap(lambda x: x).collect())
            y_pred = np.array(predicts.select(prediction_col).rdd.flatMap(lambda x: x).collect())
            probas = np.array(predicts.select("probabilidad_clase_1").rdd.flatMap(lambda x: x).collect())

            # Calcular la curva ROC
            fpr, tpr, _ = roc_curve(y_true, probas)
            plt.figure()
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            plt.show()

            # Generar matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 8))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"]
            )
            plt.xlabel("Prediction")
            plt.ylabel("Real")
            plt.title("Confusion Matrix")
            plt.show()

            results = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }

            return results

        except Exception as e:
            print(f"[ERROR] Error generating predictions and calculating metrics: {e}")
            return {"error": str(e)}
                # Configure the model based on the name
                model_mapping = {
                    'GBT': GBTClassifier,
                    'RF': RandomForestClassifier,
                    'LR': LogisticRegression,
                    'DT': DecisionTreeClassifier,
                    'NB': NaiveBayes,
                    'SVC': LinearSVC,
                    'OVR': lambda **kwargs: OneVsRest(classifier=LogisticRegression(**kwargs)),
                    'MLP': MultilayerPerceptronClassifier,
                    'FM': FMClassifier,
                    'XGB': SparkXGBClassifier,
                    'LGBM': LightGBMClassifier,
                }

                if model_name not in model_mapping:
                    raise ValueError(f"Model '{model_name}' is not implemented.")

                if model_name == 'XGB':
                    model = SparkXGBClassifier(
                        features_col=features_col,
                        label_col=target_col,
                        prediction_col="prediction",
                        **params
                    )
                elif model_name == 'LGBM':
                    model = LightGBMClassifier(
                        featuresCol=features_col,
                        labelCol=target_col,
                        objective="binary",
                        predictionCol="prediction",
                        isUnbalance=True,
                        **params
                    )
                else:
                    # Standard initialization for other models
                    model_class = model_mapping[model_name]
                    model = model_class(featuresCol=features_col, labelCol=target_col, **params)

                # Create the pipeline
                pipeline = Pipeline(stages=[model])

                # Configure the evaluator
                evaluator = BinaryClassificationEvaluator(labelCol=target_col, metricName=metric_name)

                # Configure CrossValidator
                crossval = CrossValidator(
                    estimator=pipeline,
                    evaluator=evaluator,
                    estimatorParamMaps=ParamGridBuilder().build(),
                    numFolds=3,
                    parallelism=4
                )

                # Fit the CrossValidator
                try:
                    cv_model = crossval.fit(train_df)

                    # Evaluate the model on the test set
                    predictions = cv_model.transform(test_df)
                    metric = evaluator.evaluate(predictions)

                    # Return detailed results
                    return {
                        'loss': -metric,
                        'status': STATUS_OK,
                        'model_name': model_name,
                        'params': params,
                        'metric': metric,
                        'model': cv_model,
                    }

                except Exception as e:
                    # Handle errors during training or evaluation
                    print(f"Error training model '{model_name}' with parameters {params}: {str(e)}")
                    return {
                        'loss': float('inf'),
                        'status': STATUS_FAIL,
                        'model_name': model_name,
                        'params': params,
                    }

            # Espacio de búsqueda para cada modelo
            @staticmethod
            def define_search_space():
                """
                Define un espacio de búsqueda optimizado para los modelos en model_list.
                """
                # Mapeo de modelos a sus hiperparámetros
                model_params = {
                    'GBT': {
                        'model': 'GBT',
                        'maxIter': hp.choice('gbt_maxIter', [10, 20, 50]),
                        'maxDepth': hp.choice('gbt_maxDepth', [5, 10]),
                        'stepSize': hp.uniform('gbt_stepSize', 0.1, 0.3)
                    },
                    'RF': {
                        'model': 'RF',
                        'numTrees': hp.choice('rf_numTrees', [10, 50]),
                        'maxDepth': hp.choice('rf_maxDepth', [5, 10]),
                        'minInstancesPerNode': hp.choice('rf_minInstancesPerNode', [1, 5])
                    },
                    'LR': {
                        'model': 'LR',
                        'regParam': hp.loguniform('lr_regParam', -3, -1),
                        'elasticNetParam': hp.uniform('lr_elasticNetParam', 0, 0.8)
                    },
                    'DT': {
                        'model': 'DT',
                        'maxDepth': hp.choice('dt_maxDepth', [5, 10]),
                        'minInstancesPerNode': hp.choice('dt_minInstancesPerNode', [1, 5])
                    },
                    'NB': {
                        'model': 'NB',
                        'smoothing': hp.uniform('nb_smoothing', 0.5, 1.5)
                    },
                    'SVC': {
                        'model': 'SVC',
                        'regParam': hp.uniform('svc_regParam', 0.01, 0.5),
                        'maxIter': hp.choice('svc_maxIter', [50, 100])
                    },
                    'OVR': {
                        'model': 'OVR',
                        'regParam': hp.uniform('ovr_regParam', 0.01, 0.5),
                        'elasticNetParam': hp.uniform('ovr_elasticNetParam', 0, 0.5)
                    },
                    'MLP': {
                        'model': 'MLP',
                        'layers': hp.choice('mlp_layers', [[20, 10, 5, 2]]),
                        'maxIter': hp.choice('mlp_maxIter', [50, 100])
                    },
                    'FM': {
                        'model': 'FM',
                        'stepSize': hp.uniform('fm_stepSize', 0.01, 0.3),
                        'factorSize': hp.choice('fm_factorSize', [4, 8]),
                        'regParam': hp.uniform('fm_regParam', 0.01, 0.3)
                    },
                    'XGB': {
                        'model': 'XGB',
                        'maxDepth': hp.choice('xgb_maxDepth', [3, 5]),
                        'eta': hp.uniform('xgb_eta', 0.01, 0.2),
                        'subsample': hp.uniform('xgb_subsample', 0.6, 0.9),
                        'numRound': hp.choice('xgb_numRound', [10, 50]),
                        'minChildWeight': hp.choice('xgb_minChildWeight', [1, 3]),
                        'regLambda': hp.uniform('xgb_regLambda', 0.0, 0.5),
                        'regAlpha': hp.uniform('xgb_regAlpha', 0.0, 0.5)
                    },
                    'LGBM': {
                        'model': 'LGBM',
                        'numLeaves': hp.choice('lgbm_num_leaves', [31, 63, 127]),
                        'learningRate': hp.uniform('lgbm_learning_rate', 0.01, 0.2),
                        'numIterations': hp.choice('lgbm_num_iterations', [50, 100, 200]),
                        'maxDepth': hp.choice('lgbm_max_depth', [5, 10, 15]),
                        'minDataInLeaf': hp.choice('lgbm_min_data_in_leaf', [10, 20, 50]),
                        'featureFraction': hp.uniform('lgbm_feature_fraction', 0.6, 0.9),
                        'lambdaL1': hp.uniform('lgbm_lambda_l1', 0.0, 0.5),
                        'lambdaL2': hp.uniform('lgbm_lambda_l2', 0.0, 0.5)
                    }
                }

                # Filtrar modelos según los disponibles en model_list
                space = [model_params[model] for model in model_list if model in model_params]

                if not space:
                    raise ValueError("No valid models found in model_list for search space definition.")

                return hp.choice('classifier_type', space)

            # Evaluar el número de modelos en la lista
            if len(model_list) <= 3:
                trials = Trials()
                best = None
                best_metric = float('-inf')

                try:
                    # Entrenar todos los modelos individualmente con un único `fmin`
                    for i in range(5 * len(model_list)):
                        trial = fmin(
                            fn=objective,
                            space=define_search_space(),
                            algo=tpe.suggest,
                            max_evals=i + 1,
                            trials=trials
                        )
                        
                        # Obtener el modelo actual del último trial
                        current_trial = trials.trials[-1]['result']
                        if current_trial['status'] == STATUS_OK:
                            current_metric = -current_trial['loss']
                            
                            # Comparar con el mejor modelo hasta ahora
                            if current_metric > best_metric:
                                best_metric = current_metric
                                best_model = current_trial['model']
                                best = {
                                    'model_name': current_trial['model_name'],
                                    'params': current_trial['params'],
                                    'metric': current_metric,
                                    'model': current_trial['model']
                                }
                                print(f"[INFO] New best model found: {best['model_name']} with metric: {best['metric']}")
                except StopIteration:
                    print("[INFO] Ending the search for the time limit reached.")
                except Exception as e:
                    print(f"[ERROR] Unexpected error during training: {e}")
                
            else:
                start_time = time.time()
                # Entrenar cada modelo individualmente
                individual_trials = []
                all_trials = Trials()
                
                elapsed_time = time.time() - start_time
                if elapsed_time > (time_limit/2):
                    print("[INFO] Ending the search for the time limit reached.")
                    raise StopIteration

                try:
                    fmin(
                        fn=objective,
                        space=define_search_space(),
                        algo=tpe.suggest,
                        max_evals=5 * len(model_list),
                        trials=all_trials
                    )
                    individual_trials.extend(all_trials.trials)

                except StopIteration:
                    print("[INFO] Ending the search for the time limit reached.")

                # Preparar combinaciones de modelos
                start_time = time.time()
                best_ensemble = None
                best_metric = float('-inf') 
                ensemble_results = []
                train_df.persist(StorageLevel.MEMORY_AND_DISK)

                # Obtener los top modelos de cada tipo
                top_models = generate_top_models(all_trials.trials, model_list, top_n=2)

                # Generar combinaciones con repetición de entre 2 y 5 modelos
                model_combinations = generate_combinations_with_params(top_models, n_range=(2, 3))
                try:
                    for combination in model_combinations:
                        elapsed_time = time.time() - start_time
                        if elapsed_time > (time_limit/2):
                            print("[INFO] Ending the search for the time limit reached.")
                            raise StopIteration

                        combined_models = []
                        raw_prediction_cols = []
                        probability_cols = []

                        for i, model_instance in enumerate(combination):
                            try:
                                model = model_instance['model']
                                params = model_instance['params']
                                model_name = model_instance['name']
                                model_key = model_instance['key']

                                # Añadir modelo al pipeline
                                combined_models.append(model)

                                # Renombrar columnas solo si es necesario
                                raw_prediction_cols.append(f"rawPrediction_{i}")
                                probability_cols.append(f"probability_{i}")
                                combined_models.append(
                                    AuxRenamePredictionColumn(columns_to_rename={
                                        "prediction": f"prediction_{i}",
                                        "rawPrediction": raw_prediction_cols[-1],
                                        "probability": probability_cols[-1]
                                    })
                                )
                            except StopIteration:
                                print(f"Model '{model_name}' not found in trials. Skipping combination.")
                                break
                        else:
                            # Añadir etapa de combinación si todos los modelos están presentes
                            combined_models.append(AuxCombinePredictions(
                                raw_prediction_cols=raw_prediction_cols,
                                probability_cols=probability_cols
                            ))

                            # Crear y ajustar pipeline
                            pipeline = Pipeline(stages=combined_models)
                            ensemble_model = pipeline.fit(train_df)

                            # Generar predicciones y evaluar
                            predictions = ensemble_model.transform(test_df)
                            evaluator = BinaryClassificationEvaluator(
                                labelCol=target_col,
                                rawPredictionCol="rawPrediction",
                                metricName=metric_name
                            )
                            metric = evaluator.evaluate(predictions)

                            # Agregar resultados si la métrica es válida
                            if metric > 0:
                                model_names = " + ".join([model_instance['name'] for model_instance in combination])
                                print(f"[INFO] Combination {model_names} yielded metric: {metric}")
                                ensemble_results.append({
                                    'models': model_names,
                                    'metric': metric,
                                    'model': ensemble_model
                                })
                            else:
                               print(f"[INFO] Combination {model_names} yielded invalid metric.")

                except StopIteration:
                    print("[INFO] Ending the search for the time limit reached.")
                
                finally:
                    train_df.unpersist()

                # Seleccionar el mejor ensemble
                if ensemble_results:
                    best_ensemble = max(ensemble_results, key=lambda x: x['metric'])
                    print(f"[INFO] The winning assembly is: {best_ensemble['models']} with metric: {best_ensemble['metric']}")
                    best_model = best_ensemble['model']
                    best_metric = best_ensemble['metric']
                    best = {
                        'model_name': "Ensemble",
                        'params': {},
                        'metric': best_ensemble['metric'],
                        'model': best_ensemble['model'],
                        'models_combined': best_ensemble['models']
                    }
                else:
                    raise ValueError("[ERROR] No valid ensemble models were found.")

            # Guardar el mejor modelo y métricas en MLflow
            mlflow.set_registry_uri('databricks-uc')
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():
                mlflow.log_params(best)
                mlflow.log_metric("Best Metric", best_metric)

                # Inferencia de firma del modelo
                sample_input = train_df.limit(5)
                sample_output = best_model.transform(sample_input)
                signature = infer_signature(sample_input.toPandas(), sample_output.toPandas())

                # Registrar el mejor modelo
                model_name_full = f"{catalog}.{schema}.{model_name_publish}"
                mlflow.spark.log_model(best_model, artifact_path="best_model", signature=signature)
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
                registered_model = mlflow.register_model(model_uri, name=model_name_full)

                # Crear cliente para interactuar con MLflow
                client = MlflowClient()

                # Obtener versiones existentes del modelo
                try:
                    existing_versions = client.get_registered_model(name=model_name_full).latest_versions
                    version_numbers = [int(version.version) for version in existing_versions]
                    next_version = max(version_numbers) + 1
                except Exception as e:
                    # Si no hay versiones previas, empezar desde 1
                    print(f"Error retrieving existing versions: {e}")
                    next_version = 1

                # Asignar alias a la nueva versión del modelo
                alias = f"model_version_v{next_version}"
                client.set_registered_model_alias(name=model_name_full, alias=alias, version=registered_model.version)

                print('calculating predictions...')
                # Obtener las predicciones y calcular métricas adicionales
                predictions = best_model.transform(test_df)
                y_true = predictions.select(target_col).rdd.flatMap(lambda x: x).collect()
                y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
                try:
                    if "probability" not in predictions.columns:
                        predictions = add_probability_column(predictions)
                    else:
                        probas = predictions.select("probability").rdd.map(lambda row: row['probability'][1]).collect()
                except Exception as e:
                    logging.error(f"Error retrieving the probability column: {e}")
                    probas = None

                # Calcular roc_auc
                print(f"Calculating roc_auc...")
                roc_auc = roc_auc_score(y_true, probas)

                print('calculating lift...')
                # Calcular lift para predicciones generales
                try:
                    positive_probs = [p for p, t in zip(probas, y_true) if t == 1]
                    f = gaussian_kde(positive_probs)
                    kde_all = gaussian_kde(probas)

                    lift_values = kde_positive(probas) / kde_all(probas)
                    lift = np.mean(lift_values)
                    mlflow.log_metric("lift", lift)

                except Exception as e:
                    logging.warning(f"Error calculating lift: {e}")
                    lift = None

                # Validacion del mejor humbral
                threshold_metrics = []

                for threshold in thresholds:
                    print(f"Assessing the threshold: {threshold}")
                    
                    adjusted_predictions = predictions.withColumn(
                        "adjusted_prediction",
                        F.when(vector_to_array(F.col("probability"))[1] >= threshold, 1).otherwise(0)
                    )

                    # Tomar el 10% de los datos
                    sample_fraction = 0.1
                    sampled_predictions = adjusted_predictions.sample(False, sample_fraction, seed=42)
                    
                    tp = sampled_predictions.filter(f"{target_col} = 1 AND adjusted_prediction = 1").count()
                    tn = sampled_predictions.filter(f"{target_col} = 0 AND adjusted_prediction = 0").count()
                    fp = sampled_predictions.filter(f"{target_col} = 0 AND adjusted_prediction = 1").count()
                    fn = sampled_predictions.filter(f"{target_col} = 1 AND adjusted_prediction = 0").count()

                    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else None
                    precision = tp / (tp + fp) if (tp + fp) != 0 else None
                    recall = tp / (tp + fn) if (tp + fn) != 0 else None
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision and recall) else None

                    metric = {
                        "threshold": threshold,
                        "accuracy": accuracy if accuracy is not None else 0.0,
                        "precision": precision if precision is not None else 0.0,
                        "recall": recall if recall is not None else 0.0,
                        "f1_score": f1 if f1 is not None else 0.0
                    }
                    threshold_metrics.append(metric)

                    # Registrar métricas individuales por umbral
                    mlflow.log_metric(f"accuracy_threshold_{threshold}", metric["accuracy"])
                    mlflow.log_metric(f"precision_threshold_{threshold}", metric["precision"])
                    mlflow.log_metric(f"recall_threshold_{threshold}", metric["recall"])
                    mlflow.log_metric(f"f1_score_threshold_{threshold}", metric["f1_score"])
