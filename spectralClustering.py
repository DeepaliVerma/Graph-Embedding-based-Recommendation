import pandas as pd
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
import pickle
def createPickle(df, obj_file_path):
    object_file_write = open(obj_file_path, 'wb')
    pickle.dump(df, object_file_write)
    
def returnPickle(obj_file_path):
    file_read = open(obj_file_path, 'rb')
    df_pickle = pickle.load(file_read)
    return df_pickle

G = nx.read_gml("/content/drive/My Drive/8th_Sem_Project PES_293_323_355/final_weighted_user_graph.gml")
A = nx.to_numpy_matrix(G)
clustering = SpectralClustering(affinity="precomputed",assign_labels="discretize",random_state=0).fit(A)

print(clustering.labels_)
print(len(list(G.nodes)))

clusteringX = {}
l = list(G.nodes)
for i in range(len(clustering.labels_)):
  clusteringX[l[i]] = clustering.labels_[i]
print(clusteringX)

newDict={}
for i in range(len(l)):
  ll = [k for k,v in clusteringX.items() if v == clustering.labels_[i]]
  if clustering.labels_[i] not in newDict.keys():
    newDict[clustering.labels_[i]] = ll
  else:
    newDict[clustering.labels_[i]] = list(set(newDict[clustering.labels_[i]]) | set(ll))
print("dict with count created")

user_dic_path = 'drive/My Drive/Embeddings for yelp network/embeddings/spactralClusteringDict.pkl'
createPickle(clusteringX, user_dic_path)
user_dic_pickle = returnPickle(user_dic_path)
user_dic_path1 = 'drive/My Drive/Embeddings for yelp network/embeddings/spactralClusteringDictWithCount.pkl'
createPickle(newDict, user_dic_path1)
user_dic_pickle = returnPickle(user_dic_path1)