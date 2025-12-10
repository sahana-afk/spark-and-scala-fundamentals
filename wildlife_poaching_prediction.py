# --------------------------------------------------------------
# Wildlife Poaching Prediction System - PySpark Full Ecosystem
# --------------------------------------------------------------
# Components:
# 1. Spark SQL for data analysis
# 2. MLlib for machine learning
# 3. Streaming for real-time data simulation
# 4. GraphFrames for graph analytics (GraphX alternative)
# --------------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.streaming import StreamingContext
from graphframes import GraphFrame
import time

# --------------------------------------------------------------
# 1️⃣ Initialize Spark Session
# --------------------------------------------------------------
spark = SparkSession.builder \
    .appName("Wildlife Poaching Prediction System") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

sc = spark.sparkContext
print("\n✅ Spark Session started successfully!")

# --------------------------------------------------------------
# 2️⃣ Load CSV Dataset
# --------------------------------------------------------------
file_path = "poaching_data.csv"

df = spark.read.csv(file_path, header=True, inferSchema=True)
print("\n📂 Loaded CSV Dataset:")
df.show()

# --------------------------------------------------------------
# 3️⃣ Feature Engineering + Spark SQL
# --------------------------------------------------------------
df.createOrReplaceTempView("poaching_data")

sql_result = spark.sql("""
SELECT *,
       CASE 
           WHEN temperature > 36 AND animal_density > 10 THEN 1
           ELSE 0
       END AS high_risk_zone
FROM poaching_data
""")

print("\n📊 SQL Analysis Result:")
sql_result.show()

# Assemble features for ML model
assembler = VectorAssembler(
    inputCols=["latitude", "longitude", "temperature", "animal_density"],
    outputCol="features"
)
feature_df = assembler.transform(sql_result)

# Train/Test Split
train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)

# Random Forest Classifier
rf = RandomForestClassifier(labelCol="poaching_incident", featuresCol="features", numTrees=10)
model = rf.fit(train_df)

# Predictions
predictions = model.transform(test_df)
print("\n🤖 ML Predictions:")
predictions.select("id", "features", "poaching_incident", "prediction").show()

# Model Evaluation
evaluator = BinaryClassificationEvaluator(labelCol="poaching_incident")
auc = evaluator.evaluate(predictions)
print(f"\n🎯 Model AUC (Accuracy) = {auc:.2f}")

# --------------------------------------------------------------
# 4️⃣ Spark Streaming Simulation
# --------------------------------------------------------------
print("\n⏳ Starting streaming simulation...")

ssc = StreamingContext(sc, 5)  # 5-second batch interval

def process_stream(rdd):
    if not rdd.isEmpty():
        df_stream = spark.createDataFrame(rdd, df.schema)
        print("\n⚡ Incoming Live Data Batch:")
        df_stream.show()

        # Apply the trained model on the live stream
        live_features = assembler.transform(df_stream)
        live_predictions = model.transform(live_features)
        print("\n🔍 Live Stream Predictions:")
        live_predictions.select("id", "latitude", "longitude", "prediction").show()

# Create mock streaming data
rdd_queue = []
for _ in range(3):
    new_data = df.rdd.map(lambda row: (
        row.id,
        row.latitude + float(rand().first()),
        row.longitude + float(rand().first()),
        row.temperature + float(rand().first()),
        row.animal_density + float(rand().first()),
        row.poaching_incident
    ))
    rdd_queue.append(new_data)

input_stream = ssc.queueStream(rdd_queue)
input_stream.foreachRDD(process_stream)

ssc.start()
time.sleep(10)
ssc.stop(stopSparkContext=False, stopGraceFully=True)

# --------------------------------------------------------------
# 5️⃣ GraphFrames - Zone Relationship Graph (GraphX Alternative)
# --------------------------------------------------------------
print("\n🕸️ GraphFrames: Building zone relationship graph...")

# Create vertices (zones)
vertices = spark.createDataFrame([
    ("1", "Zone A"),
    ("2", "Zone B"),
    ("3", "Zone C")
], ["id", "name"])

# Create edges (connections between zones)
edges = spark.createDataFrame([
    ("1", "2", "adjacent"),
    ("2", "3", "adjacent"),
    ("1", "3", "high-risk corridor")
], ["src", "dst", "relationship"])

# Build GraphFrame
g = GraphFrame(vertices, edges)

print("\n📍 Graph Vertices:")
g.vertices.show()

print("\n🔗 Graph Edges:")
g.edges.show()

print("\n📊 In-Degree (zones receiving connections):")
g.inDegrees.show()

print("\n📊 Out-Degree (zones connecting to others):")
g.outDegrees.show()

# --------------------------------------------------------------
# 6️⃣ End
# --------------------------------------------------------------
print("\n✅ Wildlife Poaching Prediction System completed successfully!")
spark.stop()
