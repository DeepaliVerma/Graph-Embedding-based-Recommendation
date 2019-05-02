import json
import pickle
from pprint import pprint
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  
from matplotlib import style 


def createPickle(df, obj_file_path):
    object_file_write = open(obj_file_path, 'wb')
    pickle.dump(df, object_file_write)
    
def returnPickle(obj_file_path):
    file_read = open(obj_file_path, 'rb')
    df_pickle = pickle.load(file_read)
    return df_pickle
  
with open('drive/My Drive/Embeddings for yelp network/embeddings/hope.list') as f:
    data = json.load(f)
    
cost =[] 
for i in range(1, 11): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(data) 
    cost.append(KM.inertia_)      
  
# plot the cost against K values 
plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.show() # clear the plot

#uncomment the below lines for final k means :)
KM = KMeans(n_clusters = 3,max_iter = 500) 
KM.fit(data)
#print(KM.labels_)
kmeansDict = {}
l = list(G.nodes)
for i in range(len(KM.labels_)):

  kmeansDict[l[i]] = KM.labels_[i]
print(kmeansDict)
user_dic_path = 'drive/My Drive/Embeddings for yelp network/embeddings/hopeClusteringDict.pkl'
createPickle(kmeansDict, user_dic_path)
user_dic_pickle = returnPickle(user_dic_path)


newDict={}
for i in range(len(l)):
  ll = [k for k,v in kmeansDict.items() if v == KM.labels_[i]]
  if KM.labels_[i] not in newDict.keys():
    newDict[KM.labels_[i]] = ll
  else:
    newDict[KM.labels_[i]] = list(set(newDict[KM.labels_[i]]) | set(ll))
print("dict with count created")
print(newDict[KM.labels_[0]])
user_dic_path1 = 'drive/My Drive/Embeddings for yelp network/embeddings/hopeClusteringDictWithCount.pkl'
createPickle(newDict, user_dic_path1)
user_dic_pickle = returnPickle(user_dic_path1)


