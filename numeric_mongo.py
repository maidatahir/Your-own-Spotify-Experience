import librosa
import pymongo
import numpy as np
import os
from annoy import AnnoyIndex

# Set the root directory containing subdirectories with MP3 files
root_dir = '/home/pcn/Desktop/fma/data/fma_small'

# Connect to the MongoDB database
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client['audio_features']
collection = db['mp3_features']

# Create an Annoy index with the same number of features as the MFCC array
ann_index = AnnoyIndex(len(next(iter(collection.find()))['mfcc']), 'angular')

# Iterate through the database documents and add the MFCC features to the Annoy index
for index, document in enumerate(collection.find()):
    mfcc_features = document['mfcc']
    ann_index.add_item(index, mfcc_features)

# Build the Annoy index with 10 trees
ann_index.build(10)

# Define a function to query the Annoy index for the k nearest neighbors
def get_nearest_neighbors(file_name, k=5):
    document = collection.find_one({'file_name': file_name})
    if document:
        mfcc_features = document['mfcc']
        nearest_neighbors = ann_index.get_nns_by_vector(mfcc_features, k, search_k=-1, include_distances=True)
        return [(collection.find_one({'_id': idx})['file_name'], distance) for idx, distance in zip(nearest_neighbors[0], nearest_neighbors[1])]
    else:
        return []

# Example usage: Find the 5 nearest neighbors for a given file
nearest_neighbors = get_nearest_neighbors('example.mp3')
print(nearest_neighbors)

client.close()