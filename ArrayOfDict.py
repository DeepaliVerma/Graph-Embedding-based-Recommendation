import json
with open('drive/My Drive/Embeddings for yelp network/embeddings/hope.list') as f:
    data = json.load(f)
with open('drive/My Drive/Embeddings for yelp network/embeddings/spectral.list') as f:
    data1 = json.load(f)
G = nx.read_gml("drive/My Drive/Embeddings for yelp network/final_weighted_user_graph.gml")
l = list(G.nodes)
kmeansDict = {}
spectralDict = {}
for i in range(len(l)):
  kmeansDict[l[i]] = data[i]
  spectralDict[l[i]] = data[i]
user_dic_path = 'drive/My Drive/Embeddings for yelp network/embeddings/hopeDict.pkl'
user_dic_path1 = 'drive/My Drive/Embeddings for yelp network/embeddings/spectralDict.pkl'
createPickle(kmeansDict, user_dic_path)
user_dic_pickle = returnPickle(user_dic_path)
createPickle(spectralDict, user_dic_path1)
user_dic_pickle = returnPickle(user_dic_path1)
