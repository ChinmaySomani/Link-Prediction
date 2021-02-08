import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import operator
import pandas as pd
import scipy.io as sci
from similarity_method import cn_index_final, lhn2_index_final, cbl_index_final, matrix_forest_index_final

def find_index(graph, e):
  node_name = list(graph.nodes)
  for x in range(len(node_name)):
    if(e == node_name[x]):
      return(x)

      
def graphload(filename): 
  G = nx.read_gml(filename)
  if(nx.is_directed(G)):
    G = G.to_undirected()
  #Creating data of 50% non edges and 50% edge training example and labels
  data = list(G.edges)
  non_edge = list(nx.non_edges(G))
  y = {}
  for i in data:
    y[i] = 1
  random.shuffle(non_edge)
  for i in range(0, 2126):
    data.append(non_edge[i])
    y[non_edge[i]] = 0

#Creating Panda Dataframe
  df = pd.DataFrame()
  df['Edges'] = data

  df['CN'] = df['Edges'].map(cn_index_final(G, data).get)
  df['LHN2'] = df['Edges'].map(lhn2_index_final(G, data).get)
  df['CBL'] = df['Edges'].map(cbl_index_final(G, data).get)
  df['MFI'] = df['Edges'].map(matrix_forest_index_final(G, data).get)
  df['y'] = df['Edges'].map(y.get)

  det = df.to_numpy()
  det = np.asmatrix(det)

  y_orig = det[:, 5]
  x_orig = det[:, 1:5]
# print(x_orig.shape)
  return G, df, x_orig, y_orig