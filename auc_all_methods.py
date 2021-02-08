import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import operator
import pandas as pd
import scipy.io as sci
import math
# %matplotlib inline

def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in range(n):
        if leftovers:
           extra= [ leftovers.pop() ] 
        else:
           extra= []
        yield ys[c*size:(c+1)*size] + extra

# Find the index for the adjacency Matrix
def find_index(graph, e):
  node_name = list(graph.nodes)
  for x in range(len(node_name)):
    if(e == node_name[x]):
      return(x)

#Local Index - Common Neighbour
def common_neighbor_index(folds, subs, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    accuracy = []
    aupr = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1).dot(np.array((nx.to_numpy_matrix(i) != 0) * 1))
        edgesWithScore = {}
        edges = nx.non_edges(i)
        for e in edges:
            x = find_index(i, e[0])
            y = find_index(i, e[1])
            edgesWithScore[e] = mat[x][y]
    # return edgesWithScore
        highScore = 0
        sameScore = 0
        allScore = 0
        for e in edgesWithScore:
            if e not in folds[step]:
                for s in folds[step]:
                  edgeScoreS = edgesWithScore.get(s)
                  if (edgeScoreS):
                    if edgesWithScore[e] < edgeScoreS:
                        highScore += 1
                    elif edgesWithScore[e] == edgeScoreS:
                        sameScore += 1
                    allScore += 1
        auc.append(float(highScore + 0.5*sameScore)/float(allScore))
        accuracy.append(float(sameScore + highScore)/ float(allScore))
        aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)


#Jaccard Index
def jaccard_index(folds, subs, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    accuracy = []
    aupr = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1).dot(np.array((nx.to_numpy_matrix(i) != 0) * 1))
        edgesWithScore = {}
        edges = nx.non_edges(i)
        for e in edges:
            x = find_index(i, e[0])
            y = find_index(i, e[1])
            if nx.degree(i, e[0]) != 0 or nx.degree(i, e[1]) != 0:
                edgesWithScore[e] = float(mat[x][y])/float(len(set(i[e[0]])|set(i[e[1]])))
            else:
                edgesWithScore[e] = 0
        highScore = 0
        sameScore = 0
        allScore = 0
        for e in edgesWithScore:
            if e not in folds[step]:
                for s in folds[step]:
                  edgeScoreS = edgesWithScore.get(s)
                  if(edgeScoreS):
                    if edgesWithScore[e] < edgeScoreS:
                        highScore += 1
                    elif edgesWithScore[e] == edgeScoreS:
                        sameScore += 1
                    allScore += 1
        auc.append(float(highScore + 0.5*sameScore)/float(allScore))
        accuracy.append(float(sameScore + highScore)/ float(allScore))
        aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)


# LHN1 Index
def LHN1_index(folds, subs, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    accuracy = []
    aupr = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1).dot(np.array((nx.to_numpy_matrix(i) != 0) * 1))
        edgesWithScore = {}
        edges = nx.non_edges(i)
        for e in edges:
            x = find_index(i, e[0])
            y = find_index(i, e[1])
            j = nx.degree(i, e[0])
            k = nx.degree(i, e[1])
            if j != 0 and k != 0:
                edgesWithScore[e] = float(mat[x][y])/float(j * k)
            else:
                edgesWithScore[e] = 0
        highScore = 0
        sameScore = 0
        allScore = 0
        for e in edgesWithScore:
            if e not in folds[step]:
                for s in folds[step]:
                  edgeScoreS = edgesWithScore.get(s)
                  if(edgeScoreS):
                    if edgesWithScore[e] < edgeScoreS:
                        highScore += 1
                    elif edgesWithScore[e] == edgeScoreS:
                        sameScore += 1
                    allScore += 1
        auc.append(float(highScore + 0.5*sameScore)/float(allScore))
        accuracy.append(float(sameScore + highScore)/ float(allScore))
        aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)
#Global Index - Katz Index
def katz_index(folds, subs, beta = 0, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    aupr = []
    accuracy = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
        ide = np.identity(len(mat))
        if beta == 0:
            beta = (1/float(max(np.linalg.eigh(mat)[0])))/2
            print(beta)
        sim = np.linalg.inv(ide - beta*mat) - ide
        edgesWithScore = {}
        edges = nx.non_edges(i)
        for e in edges:
            x = find_index(i, e[0])
            y = find_index(i, e[1])
            edgesWithScore[e] = sim[x][y]
        highScore = 0
        sameScore = 0
        allScore = 0
        for e in edgesWithScore:
            if e not in folds[step]:
                for s in folds[step]:
                  edgeScoreS = edgesWithScore.get(s)
                  if(edgeScoreS):
                    if edgesWithScore[e] < edgeScoreS:
                        highScore += 1
                    elif edgesWithScore[e] == edgeScoreS:
                        sameScore += 1
                    allScore += 1
        auc.append(float(highScore + 0.5*sameScore)/float(allScore))
        accuracy.append(float(sameScore + highScore)/ float(allScore))
        aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)


#LHN2 Index (Global)
def lhn2_index(folds, subs, phi = 0.5, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    aupr = []
    accuracy = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
        ide = np.identity(len(mat))
        dma = np.diagflat(mat.sum(axis = 1))
        if np.linalg.det(dma) != 0:
            lambd = float(max(np.linalg.eigh(mat)[0]))
            sim = (2 * i.number_of_edges() * lambd * np.linalg.inv(dma)).dot(np.linalg.inv(
            ide - (phi/lambd) * mat)).dot(np.linalg.inv(dma))
            edgesWithScore = {}
            edges = nx.non_edges(i)
            for e in edges:
                x = find_index(i, e[0])
                y = find_index(i, e[1])
                edgesWithScore[e] = sim[x][y]
            highScore = 0
            sameScore = 0
            allScore = 0
            for e in edgesWithScore:
                if e not in folds[step]:
                    for s in folds[step]:
                      edgeScoreS = edgesWithScore.get(s)
                      if(edgeScoreS):
                        if edgesWithScore[e] < edgeScoreS:
                            highScore += 1
                        elif edgesWithScore[e] == edgeScoreS:
                            sameScore += 1
                        allScore += 1
            auc.append(float(highScore + 0.5*sameScore)/float(allScore))
            accuracy.append(float(sameScore + highScore)/ float(allScore))
            aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)

#Cosine Based on l+
def cbl_index(folds, subs, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    aupr = []
    accuracy = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
        dma = np.diagflat(mat.sum(axis = 1))
        sim = np.linalg.pinv(dma - mat)
        edgesWithScore = {}
        edges = nx.non_edges(i)
        for e in edges:
            x = find_index(i, e[0])
            y = find_index(i, e[1])
            edgesWithScore[e] = float(sim[x][y])/np.sqrt(sim[x][x] * sim[y][y])
        highScore = 0
        sameScore = 0
        allScore = 0
        for e in edgesWithScore:
            if e not in folds[step]:
                for s in folds[step]:
                  edgeScoreS = edgesWithScore.get(s)
                  if(edgeScoreS):
                    if edgesWithScore[e] < edgeScoreS:
                        highScore += 1
                    elif edgesWithScore[e] == edgeScoreS:
                        sameScore += 1
                    allScore += 1
        auc.append(float(highScore + 0.5*sameScore)/float(allScore))
        accuracy.append(float(sameScore + highScore)/ float(allScore))
        aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)

#Matrix Forest Index
def matrix_forest_index(folds, subs, alpha = 1, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    aupr = []
    accuracy = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
        dma = np.diagflat(mat.sum(axis = 1))
        ide = np.identity(len(mat))
        sim = np.linalg.inv(ide + alpha*(dma - mat))
        edgesWithScore = {}
        edges = nx.non_edges(i)
        for e in edges:
            x = find_index(i, e[0])
            y = find_index(i, e[1])          
            edgesWithScore[e] = sim[x][y]
        highScore = 0
        sameScore = 0
        allScore = 0
        for e in edgesWithScore:
            if e not in folds[step]:
                for s in folds[step]:
                  edgeScoreS = edgesWithScore.get(s)
                  if(edgeScoreS):
                    if edgesWithScore[e] < edgeScoreS:
                        highScore += 1
                    elif edgesWithScore[e] == edgeScoreS:
                        sameScore += 1
                    allScore += 1
        auc.append(float(highScore + 0.5*sameScore)/float(allScore))
        accuracy.append(float(sameScore + highScore)/ float(allScore))
        aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)


#L3
def l3_index(folds, subs, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    aupr = []
    accuracy = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
        edgesWithScore = {}
        edges = nx.non_edges(i)
        for e in edges:
            x = find_index(i, e[0])
            y = find_index(i, e[1])
            j = nx.degree(i, e[0])
            k = nx.degree(i, e[1])
            if j != 0 and k != 0:
                edgesWithScore[e] = 1/math.sqrt(float(j * k))
            else:
                edgesWithScore[e] = 0
        highScore = 0
        sameScore = 0
        allScore = 0
        for e in edgesWithScore:
            if e not in folds[step]:
                for s in folds[step]:
                  edgeScoreS = edgesWithScore.get(s)
                  if(edgeScoreS):
                    if edgesWithScore[e] < edgeScoreS:
                        highScore += 1
                    elif edgesWithScore[e] == edgeScoreS:
                        sameScore += 1
                    allScore += 1
        auc.append(float(highScore + 0.5*sameScore)/float(allScore))
        accuracy.append(float(sameScore + highScore)/ float(allScore))
        aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)
    
# Quasi Local Index - Local Path Index
def local_path_index(folds, subs, epsilon = 0.01, symmetric = True, nfolds = 10):
    nmis = len(folds[0])
    step = 0
    auc = []
    aupr = []
    accuracy = []
    for i in subs:
        mat = np.array((nx.to_numpy_matrix(i) != 0) * 1.)
        sim = mat.dot(mat) + epsilon*mat.dot(mat.dot(mat))
        edgesWithScore = {}
        edges = nx.non_edges(i)
        for e in edges:
            x = find_index(i, e[0])
            y = find_index(i, e[1])            
            edgesWithScore[e] = sim[x][y]
        highScore = 0
        sameScore = 0
        allScore = 0
        for e in edgesWithScore:
            if e not in folds[step]:
                for s in folds[step]:
                  edgeScoreS = edgesWithScore.get(s)
                  if(edgeScoreS):
                    if edgesWithScore[e] < edgeScoreS:
                        highScore += 1
                    elif edgesWithScore[e] == edgeScoreS:
                        sameScore += 1
                    allScore += 1
        auc.append(float(highScore + 0.5*sameScore)/float(allScore))
        accuracy.append(float(sameScore + highScore)/ float(allScore))
        aupr.append(float(highScore)/ float(sameScore + highScore))
        step += 1

    return np.mean(auc), np.mean(accuracy), np.mean(aupr)

def similarity_indices(G, symmetric = True, nfolds = 10, seed = 0):
    random.seed(seed)
    folds = [i for i in chunk(G.edges(), nfolds)]
    subs = []
    for i in range(nfolds):
        graph = G.copy()
        for c in folds[i]:
            graph.remove_edge(*c)
        subs.append(graph.copy())
    cn, cn1, cn2 = common_neighbor_index(folds, subs, symmetric = symmetric)
    jaccard, j1, j2 = jaccard_index(folds, subs, symmetric = symmetric)
    lhn1, lhn11, lhn111 = LHN1_index(folds, subs, symmetric = symmetric)
    katz, k1, k2 = katz_index(folds, subs)
    lhn2, lhn21, lhn22 = lhn2_index(folds, subs)
    cbl, cbl1, cbl2 = cbl_index(folds, subs, symmetric = symmetric)
    mfi, mfi1, mfi2 = matrix_forest_index(folds, subs, symmetric = symmetric)
    localpath, local1, local2 = local_path_index(folds, subs)
    l3, l31, l32 = l3_index(folds, subs)
    aucs = {'cn' : cn, 
            'jaccard' : jaccard, 
            'lhn1' : lhn1, 
            'katz' : katz,
            'lhn2' : lhn2,
            'cbl' : cbl,
            'mfi' : mfi,
            'localpath' : localpath,
            'l3' : l3}
    accuracy = {'cn' : cn1, 
            'jaccard' : j1, 
            'lhn1' : lhn11, 
            'katz' : k1,
            'lhn2' : lhn21,
            'cbl' : cbl1,
            'mfi' : mfi1,
            'localpath' : local1,
            'l3' : l31}
    aupr = {'cn' : cn2, 
            'jaccard' : j2, 
            'lhn1' : lhn111, 
            'katz' : k2,
            'lhn2' : lhn22,
            'cbl' : cbl2,
            'mfi' : mfi2,
            'localpath' : local2,
            'l3' : l32}
    df2 = pd.DataFrame(columns = ['cn', 'jaccard', 'lhn1', 'katz', 'lhn2', 'cbl', 'mfi', 'localpath', 'l3'])
    df2.loc['auc'] = aucs
    df2.loc['accuracy'] = accuracy
    df2.loc['aupr'] = aupr
    return df2