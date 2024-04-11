import numpy as np
import networkx as nx
import scipy.sparse as sp



def prepare_graph_data(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col



def conver_sparse_tf2np(input):
    # Convert Tensorflow sparse matrix to Numpy sparse matrix
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]



def load_data(dataset_str):
    datafolder = 'processedData/'+dataset_str+"/input/"
    name1 = ['A1_10_dict.txt', 'A2_2000_10_dict.txt', 'A3_10_dict.txt']
    graph_ = {}
    for i in name1:
        f = open(datafolder + i, 'r')
        a = f.read()
        graph_[i] = eval(a)
        f.close()
    graph1 = graph_[name1[0]]
    graph2 = graph_[name1[1]]
    graph3 = graph_[name1[2]]
    features = np.load(datafolder+'features_2000_PCA.npy')
    label = np.load(datafolder+'label.npy')
    coordinates = np.load(datafolder + 'coordinates.npy')

    features = np.mat(features)

    nx_graph1 = nx.from_dict_of_lists(graph1)
    adj1 = nx.adjacency_matrix(nx_graph1)
    nx_graph2 = nx.from_dict_of_lists(graph2)
    adj2 = nx.adjacency_matrix(nx_graph2)
    nx_graph3 = nx.from_dict_of_lists(graph3)
    adj3 = nx.adjacency_matrix(nx_graph3)


    return sp.coo_matrix(adj1), sp.coo_matrix(adj2), sp.coo_matrix(adj3), features, label,coordinates

