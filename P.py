from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pymongo import MongoClient
from annoy import AnnoyIndex
import numpy as np

# Step 1: Connect to MongoDB and retrieve the data
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']
collection = db['your_collection_name']

data_from_mongo = list(collection.find())

# Step 2: Prepare the data for training
spark = SparkSession.builder \
    .appName("MusicRecommendation") \
    .getOrCreate()

# Convert MongoDB data to DataFrame
rows = [(doc['_id'], doc['file_name'], doc['file_path'], np.array(doc['mfcc'])) for doc in data_from_mongo]
df = spark.createDataFrame(rows, ["id", "file_name", "file_path", "mfcc"])

# Flatten the mfcc arrays into a single feature vector
mfcc_dim = len(rows[0][3])
assembler = VectorAssembler(inputCols=["mfcc"], outputCol="features")
df = assembler.transform(df)

# Step 3: Train a recommendation model
als = ALS(
    maxIter=10,
    regParam=0.01,
    userCol="id",
    itemCol="id",
    ratingCol="features",  # Using features as ratings for ALS
    coldStartStrategy="drop"
)

model = als.fit(df)

# Step 4: Build ANN index for item similarity
item_embeddings = model.itemFactors.rdd.map(lambda x: (x.id, x.features)).collect()
annoy_index = AnnoyIndex(mfcc_dim, 'euclidean')  # Assuming Euclidean distance

for item_id, embedding in item_embeddings:
    annoy_index.add_item(item_id, embedding)

annoy_index.build(10)  # Build the index with 10 trees for faster querying

# Function to find similar items using ANN
def find_similar_items(item_id, top_k=5):
    similar_item_ids = annoy_index.get_nns_by_item(item_id, top_k)
    return similar_item_ids

similar_items = find_similar_items(item_id='your_item_id', top_k=5)
print(similar_items)

# Step 5: Perform hyperparameter tuning using cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 20, 30]) \
    .addGrid(als.maxIter, [10, 20]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
    .build()

crossval = CrossValidator(
    estimator=als,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(metricName="rmse"),
    numFolds=5
)

# Fit ALS model to the 'df' DataFrame
cvModel = crossval.fit(df)

# Get the best ALS model from cross-validation
best_model = cvModel.bestModel

# Print the best parameters found during cross-validation
print("Best rank:", best_model.rank)
print("Best maxIter:", best_model._java_obj.parent().getMaxIter())
print("Best regParam:", best_model._java_obj.parent().getRegParam())

# Stop Spark session
spark.stop()

