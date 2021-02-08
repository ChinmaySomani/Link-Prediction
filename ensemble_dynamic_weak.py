import os
import glob
import sys
import tensorflow as tf
from math import *
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from random import shuffle
from random import seed
import datetime
import collections
from random import randint
import math
import networkx as nx
from scipy.sparse.csgraph import laplacian

from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, \
    precision_score, f1_score, precision_recall_curve, accuracy_score, balanced_accuracy_score

import time
import random
from xlwt import Workbook
import xlrd


def data(m, t):
    data = open(t + ".txt")
    edgelist = map(lambda q: list(map(int, q.split())), data.read().split("\n")[:-1])
    data.close()
    maxi = 0
    mini = 10000000000
    edgelist = list(edgelist)
    for x in edgelist:
        if x[-1] > maxi:
            maxi = x[-1]
        if x[-1] < mini:
            mini = x[-1]
    min1 = mini
    w = int((maxi - mini) / m)
    edgelist.sort(key=lambda x: x[-1])
    arr = []
    i = 0
    for i in range(0, m + 1):
        arr = arr + [min1 + w * i]
    arri = []
    # print(arr)
    nodes = set()
    for i in range(0, m):
        temp = []
        for j in edgelist:
            if j[-1] >= arr[i] and j[-1] <= arr[i + 1]:
                temp += [[j[0], j[1]]]
        arri += [temp]
    # print(arri)
    # for x in arri:
    #     print(len(x))
    print("after read")
    return arri


def bfs(source, target, it, graph, visited):
    if it > 6:
        return 6
    if source == target:
        return it
    try:
        edge = set(graph.neighbors(source))
    except:
        return 6
    ans = 6
    for e in edge:
        if e not in visited:
            visited.add(e)
            visited1 = visited
            ans = min(bfs(e, target, it + 1, graph, visited1), ans)
    return ans


def gen_graph(l):
    t_graph = []
    for i in l:
        graph = nx.Graph()
        graph.add_edges_from(i)
        t_graph.append(graph)
        print(str(graph))
    return t_graph


def weak_est(t_graph, t, l, e):
    p0 = 0.5
    for t1 in range(0, t):
        try:
            if t_graph[t1].get_eid(e[0], e[1], directed=False, error=False) != -1:
                p0 = l * p0
            else:
                p0 = 1 - l * (1 - p0)
        except:
            p0 = 1 - l * (1 - p0)

    return p0


def cnt_nodes(g):
    val = -1
    for i in g:
        for j in i:
            if j[0] > val:
                val = j[0]
            if j[1] > val:
                val = j[1]
    return val


def gen_rand_edges(num, n):
    seen = set()
    x, y = randint(1, n), randint(1, n)
    t = 0
    while t < num:
        seen.add((x, y))
        t = t + 1
        x, y = randint(1, n), randint(1, n)
        while (x, y) in seen:
            x, y = randint(1, n), randint(1, n)
    seen = list(seen)
    return seen


def features(data, m, t):
    identity = " algo - weak dataset - " + str(t)

    l1 = data
    l = []
    for i in l1:
        edgelist = list(set(tuple(sorted(sub)) for sub in i))
        l.append(edgelist)

    t_graph = gen_graph(l)

    teCt = l[m - 1]
    reCt = gen_rand_edges(len(teCt), cnt_nodes(l))

    for i in reCt:
        teCt.append(i)

    teCt = list(set(tuple(sorted(sub)) for sub in teCt))

    list1 = []
    print("before graph" + str(identity))
    edges_dict = dict()
    for t1 in range(0, m - 1):
        starttime = time.time()
        print("before dictionary" + str(identity))
        sum = 0
        count = 0
        dict_count = 0
        dict_node = dict()
        G = t_graph[t1]
        for node in G.nodes:
            dict_node[node] = dict_count
            dict_count += 1
        print(len(G.edges))
        adj = nx.adjacency_matrix(G).todense()
        edge_length = len(G.edges)
        print("before local" + str(identity))
        preds_jc = nx.jaccard_coefficient(G)
        preds_pa = nx.preferential_attachment(G)
        preds_aa = nx.adamic_adar_index(G)
        preds_sp = nx.shortest_path_length(G)
        common_jc = np.zeros(shape=(len(adj), len(adj)))
        common_pa = np.zeros(shape=(len(adj), len(adj)))
        common_aa = np.zeros(shape=(len(adj), len(adj)))
        common_cn = np.zeros(shape=(len(adj), len(adj)))
        common_sp = np.zeros(shape=(len(adj), len(adj)))
        print("before opening jaccard" + str(identity))
        for u, v, z in preds_jc:
            u = dict_node[u]
            v = dict_node[v]
            common_jc[u][v] = z
            common_jc[v][u] = z
        print("before opening adamic adar" + str(identity))
        for u, v, z in preds_aa:
            u = dict_node[u]
            v = dict_node[v]
            common_aa[u][v] = z
            common_aa[v][u] = z
        print("before opening preferential attachment" + str(identity))
        for u, v, z in preds_pa:
            u = dict_node[u]
            v = dict_node[v]
            common_pa[u][v] = z
            common_pa[v][u] = z
        print("before calculating common neighbor" + str(identity))
        for u in G.nodes:
            for v in G.nodes:
                if u != v and G.has_node(u) and G.has_node(v):
                    cn_curr = len(sorted(nx.common_neighbors(G, u, v)))
                    u = dict_node[u]
                    v = dict_node[v]
                    common_cn[u][v] = cn_curr
                    common_cn[v][u] = cn_curr
        print("before shortest path" + str(identity))
        # print(type(preds_sp))
        # print(str(preds_sp))
        for first in preds_sp:
            u = first[0]
            for node in G.nodes:
                if node in first[1]:
                    v = node
                    if u != v and G.has_node(u) and G.has_node(v):
                        z = first[1][v]
                        # if z!=0: print("sp success")
                        u = dict_node[u]
                        v = dict_node[v]
                        common_sp[u][v] = z
                        common_sp[v][u] = z
        print("before edges after graph" + str(identity))
        for e in teCt:
            i = e[0]
            j = e[1]
            edge_key = str(e[0]) + "+" + str(e[1])
            count += 1
            print(str(count) + " - " + str(len(teCt)) + " t1 = " + str(t1) + str(identity))
            # print(e)
            list2 = []
            curr_tuple = np.zeros(7)
            if G.has_node(i) and G.has_node(j):
                i_orig = i
                j_orig = j
                i = dict_node[i]
                j = dict_node[j]
                ##aa
                aa = common_aa[i][j]
                sum += aa
                list2.append(aa)
                ##jc
                jc = common_jc[i][j]
                sum += jc
                list2.append(jc)
                ##pa
                pa = common_pa[i][j]
                sum += pa
                list2.append(pa)
                ##cn
                cn = common_cn[i][j]
                sum += cn
                list2.append(cn)
                # shortest path
                sp = common_sp[i][j]
                sum += sp
                list2.append(sp)

                print("success" + str(identity))
            else:
                # print("out of bound")
                for i in range(5):
                    list2.append(0)

            if t1 == m - 2:
                list2.append(weak_est(t_graph, m - 1, 0.5, e))
                if t_graph[m - 1].has_edge(e[0], e[1]):
                    list2.append(1)
                else:
                    list2.append(0)

            if edge_key in edges_dict:
                # print(list2)
                # print("old value = "+str(edges_dict[edge_key]))
                temp_list = list(edges_dict[edge_key])
                temp_list = temp_list + list2
                edges_dict[edge_key] = temp_list
                # print("new value = " + str(edges_dict[edge_key]))
            else:
                edges_dict[edge_key] = list2
            '''print("list 2 = ")
            print(list2)
            print(len(list2))
            if sum == 0:
                print("global problem")
            else:
                print("global resolve")'''
        endtime = time.time()
        currentDT = datetime.datetime.now()
        print(str(currentDT))
        file_all = open('./result_e_d/current_all.txt', 'a')
        text_final = str(identity) + " slice = " + str(t1) + " time = " + \
                     str((endtime - starttime)) + " date_time = " + str(currentDT) + "\n"
        file_all.write(text_final)
        print(text_final)
        file_all.close()
    for e in teCt:
        edge_key = str(e[0]) + "+" + str(e[1])
        list1.append(list(edges_dict[edge_key]))
    f = np.array(list1)
    #print("list1 = " + str(list1))
    print("after get feature" + str(identity))
    print("shape of feature = " + str(f.shape))
    return f


def get_feature(data, m, t):
    return features(data, m, t)


def make_train_test(m, t, dim):
    D = get_feature(data(m, t), m, t)
    s_train = int(0.8 * D.shape[0])
    s_test = D.shape[0] - s_train
    y_train = D[0:s_train, -1]
    x_train = D[0:s_train, 0:dim]
    x_test = D[s_train + 1:-1, 0:dim]
    y_test = D[s_train + 1:-1, -1]
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    print(y_train.shape, x_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def run_model(model,x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=True)
    test_loss, test_acc, AUC, Precision, Recall = model.evaluate(x_test, y_test, verbose=2)
    test_pred = model.predict(x_test)
    prec_per, recall_per, threshold_per = precision_recall_curve(y_test, test_pred)
    prec_per = prec_per[::-1]
    recall_per = recall_per[::-1]
    aupr_value = np.trapz(prec_per, x=recall_per)
    avg_prec_value = average_precision_score(y_test, test_pred)
    test_pred_label = np.copy(test_pred)
    a = np.mean(test_pred_label)

    for i in range(len(test_pred)):
        if test_pred[i] < a:
            test_pred_label[i] = 0
        else:
            test_pred_label[i] = 1
    acc_score_value = accuracy_score(y_test, test_pred_label)
    bal_acc_score_value = balanced_accuracy_score(y_test, test_pred_label)
    f1_value = f1_score(y_test, test_pred_label)

    print('\nTest accuracy:' + str(test_acc) + "\nAUC:" + str(AUC) + "\nPrecision:" + str(Precision) +
          "\nRecall:" + str(Recall) + "\nAUPR:" + str(aupr_value) + "\nAvgPrecision" + str(avg_prec_value) +
          "\nAccScore:" + str(acc_score_value) + "\nBalAccScore:" + str(bal_acc_score_value) + "\nF1:" + str(f1_value))

    return [aupr_value, Recall, AUC, avg_prec_value, acc_score_value, bal_acc_score_value, f1_value, Precision]

def ensemble_dymanic_weak_actual(dataset, dim=21):
    initializer = tf.keras.initializers.he_normal()

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(dim,)),
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)
    ])

    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001, amsgrad=False, name='Adam'
    )

    model.compile(optimizer=adam,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    x_train, y_train, x_test, y_test = make_train_test(5, dataset, dim)
    starttime = time.time()
    value = run_model(model, x_train, y_train, x_test, y_test)
    endtime = time.time()
    currentDT = datetime.datetime.now()
    print(str(currentDT))
    file_all = open('./result_e_d/current_all.txt', 'a')
    text_final = " model for algo g in time = " + \
                 str((endtime - starttime)) + " date_time = " + str(currentDT) + "\n"
    file_all.write(text_final)
    print(text_final)
    file_all.close()

    file_write_name = './result_e_d/result_weak/' + dataset + ".txt"
    os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
    # Workbook is created
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)
    sheet1.write(0, 0, 'Dataset')
    sheet1.write(0, 1, 'AUPR')
    sheet1.write(0, 2, 'RECALL')
    sheet1.write(0, 3, 'AUC')
    sheet1.write(0, 4, 'AVG PRECISION')
    sheet1.write(0, 5, 'ACCURACY SCORE')
    sheet1.write(0, 6, 'BAL ACCURACY SCORE')
    sheet1.write(0, 7, 'F1 MEASURE')
    sheet1.write(0, 8, 'PRECISION')
    sheet1.write(1, 0, str(dataset))
    sheet1.write(1, 1, str(value[0]))
    sheet1.write(1, 2, str(value[1]))
    sheet1.write(1, 3, str(value[2]))
    sheet1.write(1, 4, str(value[3]))
    sheet1.write(1, 5, str(value[4]))
    sheet1.write(1, 6, str(value[5]))
    sheet1.write(1, 7, str(value[6]))
    sheet1.write(1, 8, str(value[7]))

    wb.save('./result_e_d/result_weak/' + dataset + ".xls")

    tf.keras.backend.clear_session()

    return 1

def ed_control_weak(dataset_array = ["Eu-core","CollegeMsg","mathoverflow"]):
    file_write_name = './result_e_d/current_all.txt'
    os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
    for dataset in dataset_array:
        tf.keras.backend.clear_session()
        starttime = time.time()
        print("running weak for dataset = "+str(dataset))
        status = ensemble_dymanic_weak_actual(dataset)
        endtime = time.time()
        currentDT = datetime.datetime.now()
        print(str(currentDT))
        file_all = open('./result_e_d/current_all.txt', 'a')
        text_final = "full algo = weak file name = " + dataset + " time = " + \
                     str((endtime - starttime)) + " date_time = " + str(currentDT) + "\n"
        file_all.write(text_final)
        print(text_final)
        file_all.close()

ed_control_weak()


