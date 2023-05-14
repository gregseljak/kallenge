
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import networkx as nx
import time
from tqdm import tqdm
from scipy.sparse import csr_array
from operator import itemgetter

def common_subgraph_vectorization(graphs, max_num_node) :
    """ 
    vectorizes the graphs in the list 'graphs' by the subgraphs vectorization in the same base. It is important to vectorize them in the same base simultaneously 
    to have meaningful operations 

    returns :
    an array where every line is a vector representing a graph. The order is concerved 
    """
    print("start common vectorization at ", time.time())
    # graph_partial_vectorization_fast = np.vectorize(graph_partial_vectorization, excluded=['max_num_nodes'])
    graphs_dict = [graph_partial_vectorization(G , max_num_node) for G in tqdm(graphs) ]
    ALL_subgraphs = set()
    for diction in graphs_dict :
        ALL_subgraphs.update(set(diction.keys()))
    
    # for x in ALL_subgraphs.copy() :
    #     if x[0] > max_num_node :
    #         ALL_subgraphs.remove(x)

    global_dimension = len(ALL_subgraphs)
    # ALL_subgraphs = dict.fromkeys(ALL_subgraphs, range(global_dimension) )


    ALL_subgraphs = list(ALL_subgraphs)
    ALL_subgraphs.sort(key = lambda x : x[0])
    
    indexes_of_change_size_subgraphs = []
    cache_size_precedent = -1
    for i in range(len(ALL_subgraphs)) :
        if ALL_subgraphs[i][0] != cache_size_precedent :
            indexes_of_change_size_subgraphs.append(i)
        cache_size_precedent = ALL_subgraphs[i][0]
        print("SOULD BE SORTED ", ALL_subgraphs[i])


    ALL_subgraphs = dict(zip(ALL_subgraphs,range(global_dimension)))
    # each row is a molecule 
    row_array = csr_array((len(graphs_dict), global_dimension), dtype = np.int)


    for i in range(row_array.shape[0]) :
        indexes = itemgetter(*(graphs_dict[i].keys() ))(ALL_subgraphs)
        row_array[i,indexes] = list(graphs_dict[i].values())
    
    # data_frames = [ pd.DataFrame.from_dict(subgraph_hash_dict, orient='index') for subgraph_hash_dict in graphs_dict ]
    # df_total = pd.concat(data_frames, axis=1).fillna(0)
    # # columns_array = np.array(df_total)
    # print("end common vectorization at ", time.time())
    # return( columns_array.T )

    print("end common vectorization at ", time.time())
    return(row_array, indexes_of_change_size_subgraphs)




def graph_partial_vectorization(G,max_num_node) :
    """
    this function doesn't actually vectorize, it returns a dictionnary of (key : value) pairs (hash_of_subgraph : multiplicity)
    """

    subgraphs = get_all_connected_subgraphs(G, max_size =max_num_node )
    subgraph_hash = get_hash_dict_subgraphs(subgraphs)
    result = from_list_to_dict_and_multiplicity(subgraph_hash)

    return(result)


# def get_all_connected_subgraphs(G, max_size):
#     """Get all connected subgraphs by a recursive procedure, max_size is the maximum number of nodes allowed"""

#     con_comp = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    
#     def recursive_local_expand(node_set, possible, excluded, results, max_size):
#         """
#         Recursive function to add an extra node to the subgraph being formed
#         """
#         results.append(node_set)
#         if len(node_set) == max_size:
#             return
#         for j in possible - excluded:
#             new_node_set = node_set | {j}
#             excluded = excluded | {j}
#             new_possible = (possible | set(G.neighbors(j))) - excluded
#             recursive_local_expand(new_node_set, new_possible, excluded, results, max_size)
    
#     results = []
#     for cc in con_comp:
#         print(cc)
#         excluded = set()

#         for i in cc:
#             excluded.add(i)
#             recursive_local_expand({i}, set(G.neighbors(i)) - excluded, excluded, results, max_size) # min(max_size,G.size() - len(excluded)) )

#     results.sort(key=len)

#     subgraphs = [ G.subgraph(results[i]) for i in range(len(results)) ]
#     return subgraphs 

def get_all_connected_subgraphs(G, max_size):
    """Get all connected subgraphs by a recursive procedure, max_size is the maximum number of nodes allowed"""
    
    def recursive_local_expand(node_set, possible, excluded, results, max_size):
        """
        Recursive function to add an extra node to the subgraph being formed
        """
        results.append(node_set)
        if len(node_set) == max_size:
            return
        for j in possible - excluded:
            new_node_set = node_set | {j}
            excluded = excluded | {j}
            new_possible = (possible | set(G.neighbors(j))) - excluded
            recursive_local_expand(new_node_set, new_possible, excluded, results, max_size)
    
    results = []
 
    excluded = set()

    for i in G:
        excluded.add(i)
        recursive_local_expand({i}, set(G.neighbors(i)) - excluded, excluded, results, max_size) # min(max_size,G.size() - len(excluded)) )

    results.sort(key=len)


    subgraphs = [ G.subgraph(results[i]) for i in range(len(results)) ]
    return subgraphs 

def get_hash_dict_subgraphs(subgraphs) :

    return [(subgraph.number_of_nodes(),nx.weisfeiler_lehman_graph_hash( subgraph, edge_attr="labels", node_attr="labels", iterations=3, digest_size=16)) for subgraph in subgraphs]

def from_list_to_dict_and_multiplicity(subgraph_hash):
    dict = {}
    for x in subgraph_hash:
        cnt = subgraph_hash.count(x)
        dict[x] = cnt 
    
    return(dict)



def gauss_kernel( vect_a, vect_b, sigma = 1.0 ) :
    return( np.exp( -np.sum( (vect_a - vect_b)**2/(2.0*sigma) ) ) )



def subgraph_kernel(subgraph_hash_dict1,subgraph_hash_dict2 ) :
    # we have difficult time defining a kernel here, a lot of decisions are arbitrary (for instance, metaparameters)
    # we choose 
    
    df1 = pd.DataFrame.from_dict(subgraph_hash_dict1, orient='index')
    df2 = pd.DataFrame.from_dict(subgraph_hash_dict2, orient='index')
    df3 = pd.concat([df1, df2], axis=1).fillna(0)
    columns_array = np.array(df3)
    
    result = gauss_kernel(columns_array[:,0],columns_array[:,1] )

    return(result)