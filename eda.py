# Importaciones de bibliotecas estándar de Python
import datetime
import logging
import os
import math
from datetime import timedelta
from functools import reduce
from itertools import combinations
import tempfile
import shap
import yaml

# Bibliotecas de manejo de datos
import pandas as pd
import numpy as np
import geopandas as gpd
import missingno as msno
import featuretools as ft

# Bibliotecas de visualización
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

# Bibliotecas de geolocalización
from geopy.geocoders import Nominatim
from shapely.geometry import Point
import requests

# Bibliotecas de Spark SQL y tipos de datos
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Bibliotecas de Spark ML
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes, MultilayerPerceptronClassifier
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkStandardScaler, MinMaxScaler as SparkMinMaxScaler, DenseVector
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.stat import Summarizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array
from xgboost.spark import SparkXGBClassifier

# Bibliotecas de análisis de datos y modelado
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore, gaussian_kde
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay

# Bibliotecas de MLflow
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

class Eda:
    def __init__(self) -> str:
        """
        Inicializa la clase Eda.
        """
        self.tables = []
        self.dates = []
        
    @staticmethod
    def get_sample(self, df: F.DataFrame, sample_size: int, stratify_col: str = None) -> F.DataFrame:
        """
        Obtiene una muestra del DataFrame, con opción de estratificar.
        
        Parameters:
        df (F.tDataFrame): DataFrame de Spark a muestrear.
        sample_size (int): Tamaño de la muesra.
        stratify_col (str, opcional): Columna para estratificar la muestra.
        
        Returns:
        F.DataFrame: Muestra del DataFrame de Spark.
        """
        if stratify_col:
            unique_values = [row[stratify_col] for row in df.select(stratify_col).distinct().collect()]
            fraction = sample_size / df.count()
            fractions = {value: fraction for value in unique_values}
            df = df.stat.sampleBy(stratify_col, fractions, seed=42)
        else:
            df = df.sample(fraction=sample_size / df.count(), seed=42)
        print(f"[INFO]Sample size obtained: {df.count()} rows")
        return df

    def plot_correlation_matrix(self, df: F.DataFrame):
        """
        Genera una matriz de correlación para las variables numéricas en el DataFrame.
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        """
        # Seleccionar las columnas numéricas
        numeric_columns = [col_name for col_name, dtype in df.dtypes if dtype in ('int', 'double', 'float', 'bigint')]
        # Convertir a Pandas y calcular la matriz de correlación
        df_numeric = df.select(*numeric_columns).toPandas().corr()
        
        # Imprimir la matriz de correlación en formato tabla
        print("Matriz de correlación (tabla):")
        print(df_numeric)
        
        # Crear el mapa de calor
        plt.figure(figsize=(17, 12))
        sns.heatmap(df_numeric, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix Between Variables")
        plt.show()

    def univariate_analysis(self, df: F.DataFrame, column_name: str):
        """
        Realiza un análisis univariado para una columna específica.
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        column_name (str): Nombre de la columna a analizar.
        """
        if column_name not in df.columns:
            print(f"[ERROR]column '{column_name}' does not exist in the DataFrame.")
            return

        column_data = df.select(column_name).toPandas()[column_name]
        
        plt.figure(figsize=(10, 6))
        if column_data.dtype in ['int64', 'float64']:
            sns.histplot(column_data, bins=30, kde=True)
            plt.title(f"Distribution of the column '{column_name}'")
            plt.xlabel(column_name)
            plt.ylabel("Frecuency")
        else:
            sns.countplot(y=column_data, order=column_data.value_counts().index)
            plt.title(f"Distribution of the column '{column_name}'")
            plt.xlabel("Frecuency")
            plt.ylabel(column_name)
        plt.show()

    def plot_missing_values(self, df: F.DataFrame):
        """
        Genera un gráfico de matriz y barra utilizando la librería missingno para visualizar los valores nulos en el DataFrame.
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        """
        df_pandas = df.toPandas()
    
        plt.figure(figsize=(12, 8))
        msno.matrix(df_pandas, sparkline=True)
        plt.title("Matrix of Null Values", fontsize=14)
        plt.show()

        plt.figure(figsize=(12, 8))
        msno.bar(df_pandas, color="skyblue", fontsize=12)
        plt.title("Bar Chart of Null Values", fontsize=14)
        plt.show()

        plt.figure(figsize=(12, 8))
        msno.heatmap(df_pandas)
        plt.title("Heatmap of Correlation of Null Values", fontsize=14)
        plt.show()

        plt.figure(figsize=(12, 8))
        msno.dendrogram(df_pandas)
        plt.title("Dendrogram of Null Values", fontsize=14)
        plt.show()

    def plot_outliers(self, df: F.DataFrame):
        """
        Muestra los outliers de las columnas numéricas usando boxplots con un máximo de 2 gráficos por fila.
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        """
        df_pandas = df.toPandas()
        numeric_columns = [col for col in df_pandas.columns if df_pandas[col].dtype in ['int64', 'float64']]
        
        # Configurar el número de filas y columnas
        num_columns = 2
        num_rows = math.ceil(len(numeric_columns) / num_columns)
        
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(10, 5 * num_rows))
        axes = axes.flatten()  # Aplanar los ejes para facilitar la indexación

        for i, col in enumerate(numeric_columns):
            sns.boxplot(y=df_pandas[col], ax=axes[i])
            axes[i].set_title(f"[INFO]Outliers in {col}")
        
        # Ocultar ejes no usados si el número de gráficos es impar
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def temporal_analysis(self, df: pd.DataFrame):
        """
        Realiza un análisis temporal para las columnas de tipo fecha en el DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame con posibles columnas de tipo fecha.
        """
        
        # Si el DataFrame es de Spark, conviértelo a pandas
        if not isinstance(df, pd.DataFrame):
            df = df.toPandas()

        # Identificar columnas de fecha
        date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

        if not date_columns:
            print("[ERROR]No date-type columns were found in the DataFrame.")
            return
        
        for col in date_columns:
            print(f"[INFO]Temporal Analysis of the Column: {col}")
            
            # Frecuencia de registros por fecha
            df[col] = pd.to_datetime(df[col], errors='coerce')  # Convertir a datetime, ignorando errores
            df['year'] = df[col].dt.year
            df['month'] = df[col].dt.to_period('M')

            # Conteo de registros por año
            yearly_counts = df.groupby('year').size()
            monthly_counts = df.groupby('month').size()

            # Gráfico de frecuencia anual
            plt.figure(figsize=(12, 5))
            yearly_counts.plot(kind='bar')
            plt.title(f"Frequency of Records by Year - {col}")
            plt.xlabel("Year")
            plt.ylabel("Number of Records")
            plt.xticks(rotation=45)
            plt.show()

            # Gráfico de frecuencia mensual
            plt.figure(figsize=(15, 5))
            monthly_counts.plot()
            plt.title(f"Frequency of Records by Month - {col}")
            plt.xlabel("Month")
            plt.ylabel("Number of Records")
            plt.xticks(rotation=45)
            plt.show()

            # Resumen estadístico de la columna de fecha
            # min_date = df[col].min()
            # max_date = df[col].max()
            # print(f"Fecha mínima en {col}: {min_date}")
            # print(f"Fecha máxima en {col}: {max_date}")
            # print(f"Cantidad de registros únicos por año:\n{yearly_counts}")
            # print(f"Cantidad de registros únicos por mes (top 12):\n{monthly_counts.head(12)}")

        # Limpiar columnas auxiliares
        df.drop(columns=['year', 'month'], inplace=True)

    def detect_high_colinearity(self, df: F.DataFrame, threshold: float = 5.0) -> list:
        """
        Detecta columnas con alta colinealidad utilizando el Factor de Inflación de Varianza (VIF).
        
        Parameters:
        df (DataFrame): DataFrame de Spark.
        threshold (float): Umbral de VIF para detectar alta colinealidad.
        
        Returns:
        list: Lista de columnas con alta colinealidad.
        """
        # Convertir el DataFrame de Spark a Pandas
        df_pandas = df.toPandas()
        
        # Seleccionar solo columnas numéricas
        numeric_columns = [col for col in df_pandas.columns if df_pandas[col].dtype in ['int64', 'float64']]
        df_numeric = df_pandas[numeric_columns].dropna()
        
        # Calcular el VIF para cada columna numérica
        vif_data = pd.DataFrame()
        vif_data['Variable'] = df_numeric.columns
        vif_data['VIF'] = [variance_inflation_factor(df_numeric.values, i) for i in range(len(df_numeric.columns))]
        
        # Filtrar las columnas con VIF mayor al umbral y mostrar el nivel de colinealidad
        high_vif_columns = vif_data[vif_data['VIF'] > threshold]
        
        if not high_vif_columns.empty:
            print(f"[WARNING]Columns with High Collinearity (VIF > {threshold}):")
            for _, row in high_vif_columns.iterrows():
                print(f"[WARNING]Columna: {row['Variable']}, VIF: {row['VIF']:.2f}")
        else:
            print("No se encontraron columnas con alta colinealidad según el umbral especificado.")
        
        # Convertir el DataFrame de alto VIF a lista de columnas
        return high_vif_columns['Variable'].tolist()
    
    def detect_high_cardinality_categoricals(self, df: F.DataFrame, threshold: int = 20) -> list:
        """
        Detecta columnas categóricas con alta cardinalidad.
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        threshold (int): Umbral de categorías únicas para detectar alta cardinalidad.
        
        Returns:
        list: Lista de columnas categóricas con alta cardinalidad.
        """
        categorical_columns = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
        high_cardinality_columns = []

        for col in categorical_columns:
            unique_count = df.select(col).distinct().count()
            if unique_count > threshold:
                high_cardinality_columns.append(col)

        if high_cardinality_columns:
            print("[WARNING]Categorical Columns with Too Many Categories:")
            for col in high_cardinality_columns:
                print(f"{col}")
        else:
            print("[INFO]No categorical columns with high cardinality were found based on the specified threshold.")
        
        return high_cardinality_columns
    
    def cluster_data(self, df: F.DataFrame, features: list, k: int = 3) -> F.DataFrame:
        """
        Realiza clustering usando KMeans y calcula el Silhouette Score.
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        features (list): Lista de nombres de columnas para el clustering.
        k (int): Número de clusters.
        
        Returns:
        F.DataFrame: DataFrame con la columna de asignación de clusters.
        """
        df_scaled = self.scale_data(df, features)
        kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol("scaled_features")
        model = kmeans.fit(df_scaled)
        df_clustered = model.transform(df_scaled)
        
        evaluator = ClusteringEvaluator(featuresCol="scaled_features", metricName="silhouette")
        silhouette = evaluator.evaluate(df_clustered)
        print(f"[INFO]Silhouette Score: {silhouette:.4f}")
        
        return df_clustered
    
    def silhouette_method(self, df: F.DataFrame, feature_cols: list, max_k: int = 10):
        """
        Encuentra el número óptimo de clusters usando el Silhouette Score.
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        feature_cols (list): Lista de nombres de columnas a usar para el clustering.
        max_k (int): Máximo número de clusters a probar.
        """
        silhouette_scores = []
        df_scaled = self.scale_data(df, feature_cols)

        for k in range(2, max_k + 1):
            kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol("scaled_features")
            model = kmeans.fit(df_scaled)
            df_clustered = model.transform(df_scaled)
            
            evaluator = ClusteringEvaluator(featuresCol="scaled_features", metricName="silhouette")
            silhouette_scores.append(evaluator.evaluate(df_clustered))
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
        plt.title("Silhouette Coefficient Method")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.show()

    def bivariate_analysis(self, df: F.DataFrame, target_column: str):
        """
        Realiza un análisis bivariado entre el target y todas las otras columnas.
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        target_column (str): Nombre de la columna target.
        """
        if target_column not in df.columns:
            print("[ERROR]The target column does not exist in the DataFrame.")
            return

        df_pandas = df.toPandas()
        for feature_column in df.columns:
            if feature_column == target_column:
                continue

            print(f"\n[INFO]Bivariate Analysis Between '{target_column}' and '{feature_column}':")

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            if df_pandas[feature_column].dtype in ['int64', 'float64']:
                sns.boxplot(x=target_column, y=feature_column, data=df_pandas, ax=axes[0])
                sns.scatterplot(x=feature_column, y=target_column, data=df_pandas, alpha=0.5, ax=axes[1])
                sns.histplot(data=df_pandas, x=feature_column, hue=target_column, kde=True, element="step", stat="density", common_norm=False, ax=axes[2])
            else:
                contingency_table = pd.crosstab(df_pandas[target_column], df_pandas[feature_column])
                print(f"Contingency Table Between {target_column} y {feature_column}:\n", contingency_table)
                sns.countplot(x=feature_column, hue=target_column, data=df_pandas, ax=axes[0])
                axes[1].axis('off')
                axes[2].axis('off')

            plt.tight_layout()
            plt.show()

    def non_linear_correlation(self, df: F.DataFrame, method: str = "spearman"):
        """
        Calcula la correlación no lineal entre variables usando Spearman o MIC (Maximal Information Coefficient).
        
        Parameters:
        df (F.DataFrame): DataFrame de Spark.
        method (str): Método de correlación a usar, "spearman" o "mic".

        Returns:
        pd.DataFrame: Una matriz de correlación no lineal.
        """
        df_pandas = df.toPandas()
        if method == "spearman":
            correlation_matrix = df_pandas.corr(method='spearman')
        elif method == "mic":
            mine = MINE(alpha=0.6, c=15)
            columns = df_pandas.columns
            mic_matrix = pd.DataFrame(index=columns, columns=columns)
            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns):
                    if i == j:
                        mic_matrix.loc[col1, col2] = 1.0
                    elif pd.isnull(mic_matrix.loc[col1, col2]):
                        mine.compute_score(df_pandas[col1], df_pandas[col2])
                        mic_matrix.loc[col1, col2] = mic_matrix.loc[col2, col1] = mine.mic()
            correlation_matrix = mic_matrix.astype(float)
        else:
            raise ValueError("[ERROR]Unrecognized method. Use 'spearman' or 'mic'.")
        
        plt.figure(figsize=(17, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(f"[SUCCESS]Non-Linear Correlation Matrix ({method.capitalize()})")
        plt.show()
        
        return correlation_matrix
    
    def add_city_coordinates(self, df: F.DataFrame, api_key: str) -> pd.DataFrame:
        """
        Obtiene las coordenadas de las ciudades usando la API de OpenCage y las une con el DataFrame.

        Parameters:
        df (F.DataFrame): DataFrame de Spark con la columna de ciudad.
        api_key (str): Clave de API para OpenCage Geocoding.

        Returns:
        pd.DataFrame: DataFrame con las coordenadas agregadas.
        """
        # Convertimos el DataFrame de Spark a Pandas
        df_pandas = df.select("CIUDAD").distinct().toPandas()  # Obtener ciudades únicas

        # Función interna para llamar a la API de OpenCage
        def get_coordinates(city_name, country="Colombia"):
            url = f"https://api.opencagedata.com/geocode/v1/json?q={city_name},{country}&key={api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                results = response.json().get('results')
                if results:
                    geometry = results[0]['geometry']
                    return geometry['lng'], geometry['lat']
            return None, None

        # Obtener coordenadas para cada ciudad única
        coordenadas = []
        for ciudad in df_pandas["CIUDAD"]:
            lng, lat = get_coordinates(ciudad)
            coordenadas.append({"CIUDAD": ciudad, "longitud": lng, "latitud": lat})

        # Convertir en DataFrame y unir con el DataFrame principal
        coordenadas_df = pd.DataFrame(coordenadas)
        df_pandas = df.toPandas()
        df_pandas = df_pandas.merge(coordenadas_df, on="CIUDAD", how="left")
        
        # Crear columna de geometría con las coordenadas
        df_pandas["geometry"] = df_pandas.apply(lambda row: Point(row["longitud"], row["latitud"]) 
                                                if pd.notnull(row["longitud"]) and pd.notnull(row["latitud"]) 
                                                else None, axis=1)
        return df_pandas
    
    def plot_city_month_grid(self, df: pd.DataFrame, shapefile_path: str):
        """
        Genera una cuadrícula de mapas que muestra el conteo de registros por ciudad en Colombia para cada mes.

        Parameters:
        df (pd.DataFrame): DataFrame con las columnas CIUDAD, FECHA_CORTE y geometry.
        shapefile_path (str): Ruta del archivo shapefile de Colombia.
        """
        
        # Si el DataFrame es de Spark, conviértelo a pandas
        if not isinstance(df, pd.DataFrame):
            df = df.toPandas()
        
        # Asegúrate de que las coordenadas están configuradas como geometría
        if 'geometry' not in df.columns or df['geometry'].isnull().any():
            df = gpd.GeoDataFrame(df, geometry=df.apply(lambda row: Point(row['longitud'], row['latitud']), axis=1))

        if 'geometry' not in df.columns:
            print("La columna 'geometry' no está en el DataFrame.")
            return
        
        # Cargar el mapa de Colombia
        colombia_map = gpd.read_file(shapefile_path)

        # Obtener meses únicos en los datos
        unique_months = sorted(df['FEC_PROCESO'].astype(int).unique())
        num_months = len(unique_months)

        # Crear la cuadrícula de gráficos
        fig = plt.figure(figsize=(15, 10))
        grid = ImageGrid(fig, 111, nrows_ncols=(3, 4), axes_pad=0.5, label_mode="L", share_all=True)
        
        for ax, month in zip(grid, unique_months):
            month_data = df[df['FEC_PROCESO'].astype('int') == month]

            # Verificar si month_data está vacío
            if month_data.empty:
                print(f"No hay datos para el mes {month}.")
                continue
            
            # Crear el mapa base de Colombia
            colombia_map.plot(ax=ax, color='lightgrey', edgecolor='black')
            
            # Agrupar sin eliminar geometry
            month_data_counts = month_data.groupby('CIUDAD').size().reset_index(name='counts')

            # Unir los datos de geometría después de agrupar
            month_data = month_data[['CIUDAD', 'geometry']].drop_duplicates().merge(month_data_counts, on='CIUDAD')

            # Graficar las ciudades en el mes actual
            # .apply(lambda x: max(2, x / 10))
            month_data.plot(ax=ax, markersize=month_data['counts'], 
                    color='blue', alpha=0.3)

            ax.set_title(f"Distribution of Counts - {month}")
            ax.axis('off')  # Ocultar ejes para mejorar la visualización

        plt.suptitle("Distribution of Counts for City in Colombia for Month", fontsize=16)
        plt.tight_layout()
        plt.show()

    def count_nulls(self, df: F.DataFrame) -> F.DataFrame:
        """
        Cuenta la cantidad de valores nulos en cada columna de un DataFrame de Spark.
        
        Parameters:
        - df: DataFrame de Spark a analizar.
        
        Returns:
        - DataFrame con el conteo de valores nulos por columna.
        """
        # Crear una lista de expresiones para contar los nulos en cada columna
        null_counts = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
        
        # Aplicar el conteo de nulos
        nulls_df = df.select(*null_counts)
        
        # Transponer el resultado para mejor legibilidad
        nulls_df = (nulls_df
                    .withColumn("row_id", F.lit(1))
                    .selectExpr("stack(" + str(len(df.columns)) + ", " +
                                ", ".join([f"'{c}', {c}" for c in df.columns]) +
                                ") as (Column, NullCount)")
                    .where(F.col("NullCount") > 0)  # Filtrar solo las columnas con valores nulos
                    .orderBy(F.desc("NullCount")))
        
        return nulls_df