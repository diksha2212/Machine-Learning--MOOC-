## IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.sparse import csr_matrix

##LOADING DATA

wiki = pd.read_csv("people_wiki.csv")

###loading word counts from file ##
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)

word_count = load_sparse_csr('E:\ML-data\Clustering\people_wiki_word_count.npz')

## Finding nearest neighbours using word counts ##

from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)

print wiki[wiki['name'] == 'Barack Obama']

## index for obama is 35817

##running knearest neighbours for barack obama

distances, indices = model.kneighbors(word_count[35817], n_neighbors=10) # 1st arg: word count vector

neighbours=pd.DataFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.merge(neighbours).sort('distance')[['id','name','distance']]

## USING THE TFIDF APPROACH ###

tf_idf = load_sparse_csr('E:\ML-data\Clustering\people_wiki_tf_idf.npz')

model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)

distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)

neighbours=pd.DataFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.merge(neighbours).sort('distance')[['id','name','distance']]
