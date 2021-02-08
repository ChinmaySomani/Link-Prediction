import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#Similarity Methods Index = Common Neighbour, LHN2, Cosine Based on L+, Matrix Forest Index

def find_index(graph, e):
  node_name = list(graph.nodes)
  for x in range(len(node_name)):
    if(e == node_name[x]):
      return(x)

      
def cn_index_final(i, data):
    mat = np.array((nx.to_numpy_matrix(i) != 0) * 1).dot(np.array((nx.to_numpy_matrix(i) != 0) * 1))
    index = {}
    for e in data:
        index[e] = len(list(nx.common_neighbors(i, e[0], e[1])))
    return index 

def lhn2_index_final(i, data, phi = 0.1):
    mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
    ide = np.identity(len(mat))
    dma = np.diagflat(mat.sum(axis = 1))
    index = {}
    if np.linalg.det(dma) != 0:
        lambd = float(max(np.linalg.eigh(mat)[0]))
        sim = (2 * i.number_of_edges() * lambd * np.linalg.inv(dma)).dot(np.linalg.inv(
        ide - (phi/lambd) * mat)).dot(np.linalg.inv(dma))
        for e in data:
            x = find_index(i, e[0])
            y = find_index(i, e[1])               
            index[e] = sim[x][y]
    return index

def cbl_index_final(i, data):
    mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
    dma = np.diagflat(mat.sum(axis = 1))
    sim = np.linalg.pinv(dma - mat)
    index = {}
    for e in data:
        x = find_index(i, e[0])
        y = find_index(i, e[1])   
        if sim[x][x] * sim[y][y] != 0:
            index[e] = float(sim[x][y])/np.sqrt(sim[x][x] * sim[y][y])
        else:
            index[e] = 0
    return index

def matrix_forest_index_final(i, data, alpha = 1):
    mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
    dma = np.diagflat(mat.sum(axis = 1))
    ide = np.identity(len(mat))
    sim = np.linalg.inv(ide + alpha*(dma - mat))
    index = {}
    for e in data:
        x = find_index(i, e[0])
        y = find_index(i, e[1])   
        index[e] = sim[x][y]
    return index