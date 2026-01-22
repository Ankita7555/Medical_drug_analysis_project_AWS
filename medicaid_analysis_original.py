# Install required packages (for EMR Notebooks or Databricks)
sc.install_pypi_package("boto3")
sc.install_pypi_package("matplotlib")
sc.install_pypi_package("seaborn")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# --- Spark session ---
spark = SparkSession.builder.appName("MedicaidDrugAnalysis").getOrCreate()

# --- Load data from S3 ---
df = spark.read.csv("s3://finalprojectankita/Input/SDUD2023.csv", header=True, inferSchema=True)

# --- Clean column names and values ---
for col in df.columns:
    df = df.withColumnRenamed(col, col.strip())

df_clean = df.withColumn("Product Name", F.trim(F.regexp_replace("Product Name", r"[^a-zA-Z0-9\s]", "")))

for col in [
    "Units Reimbursed", "Number of Prescriptions",
    "Total Amount Reimbursed", "Medicaid Amount Reimbursed", "Non Medicaid Amount Reimbursed"
]:
    df_clean = df_clean.withColumn(col, F.col(col).cast("float"))

df_clean = df_clean.dropna(subset=["State", "Product Name", "Total Amount Reimbursed"])



# --- Top 5 Drugs by State ---
window_spec = Window.partitionBy("State").orderBy(F.desc("Total Amount Reimbursed"))
df_ranked = df_clean.withColumn("rank", F.row_number().over(window_spec))
top5_df = df_ranked.filter(F.col("rank") <= 5).drop("rank")

# --- Regression Model ---
state_indexer = StringIndexer(inputCol="State", outputCol="StateIndex")
drug_indexer = StringIndexer(inputCol="Product Name", outputCol="DrugIndex")
assembler = VectorAssembler(
    inputCols=["StateIndex", "DrugIndex", "Units Reimbursed", "Number of Prescriptions"],
    outputCol="features"
)
lr = LinearRegression(featuresCol="features", labelCol="Total Amount Reimbursed")
pipeline = Pipeline(stages=[state_indexer, drug_indexer, assembler, lr])

train_data, test_data = top5_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

evaluator = RegressionEvaluator(
    labelCol="Total Amount Reimbursed", predictionCol="prediction", metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f" RMSE (Root Mean Squared Error): {rmse:.2f}")

# --------------------------------------
# 📊 VISUALIZATION SECTION
# --------------------------------------

# Convert Spark to Pandas for plotting
pandas_df = df_clean.toPandas()

# Plot 1: Top 10 Drugs by Total Reimbursement
top10_drugs = pandas_df.groupby("Product Name")["Total Amount Reimbursed"].sum().nlargest(10)
plt.figure(figsize=(10, 6))
top10_drugs.sort_values().plot(kind="barh", color="skyblue")
plt.title("Top 10 Drugs by Total Reimbursement")
plt.xlabel("Total Amount Reimbursed")
plt.ylabel("Product Name")
plt.tight_layout()
plt.show()

# Save locally first
plt.savefig("/tmp/top10_drugs.png")

# Upload to S3
s3 = boto3.client("s3")
s3.upload_file("/tmp/top10_drugs.png", "finalprojectankita", "Output/top10_drugs.png")



# Plot 2: Total Reimbursement by State
plt.figure(figsize=(12, 6))
state_totals = pandas_df.groupby("State")["Total Amount Reimbursed"].sum().sort_values(ascending=False)
state_totals.plot(kind="bar", color="salmon")
plt.title("Total Reimbursement per State")
plt.ylabel("Total Amount Reimbursed")
plt.xlabel("State")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save locally first
plt.savefig("/tmp/1top10_drugs.png")

# Upload to S3
s3 = boto3.client("s3")
s3.upload_file("/tmp/1top10_drugs.png", "finalprojectankita", "Output/1top10_drugs.png")



# Plot 3: Heatmap of Correlation
plt.figure(figsize=(8, 6))
numeric_cols = pandas_df[[
    "Units Reimbursed", "Number of Prescriptions",
    "Total Amount Reimbursed", "Medicaid Amount Reimbursed", "Non Medicaid Amount Reimbursed"
]]
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Reimbursement Metrics")
plt.tight_layout()

# Save locally before showing
plt.savefig("/tmp/correlation_heatmap_reimbursement_metrics.png")

# Upload to S3
s3 = boto3.client("s3")
s3.upload_file(
    "/tmp/correlation_heatmap_reimbursement_metrics.png",
    "finalprojectankita",
    "Output/correlation_heatmap_reimbursement_metrics.png"
)

# Show plot (only if supported in environment)
plt.show()





# Write the transformed top5 dataframe to S3 output folder in Parquet
top5_df.write.mode("overwrite").parquet("s3://finalprojectankita/Output/top5_drugs_by_state")

# Optionally, write predictions as CSV
predictions.select("State", "Product Name", "prediction", "Total Amount Reimbursed") \
    .write.mode("overwrite").option("header", True) \
    .csv("s3://your-output-bucket/output/predictions")
