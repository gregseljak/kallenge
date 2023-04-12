import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy import sparse
import networkx as nx
from molecview import display_molec

train = pkl.load(open("./training_data.pkl", "rb"))
trainy = pkl.load(open("./training_labels.pkl", "rb"))
nodelen = np.zeros(len(train))
for i in range(len(nodelen)):
    nodelen[i]=len(train[i].nodes)


def index_by_label(molecule):
    # nested list of indices with the structure
    # idxs[k] : returns list of indexes with label k
    idxs = []
    for i in range(list(molecule._node.keys())[-1]+1):
        atom = molecule._node[i]
        label = (atom["labels"])[0]
        while len(idxs)<=label:
            idxs.append([])
        idxs[label].append(i)
    return idxs

ibl = index_by_label(train[51])
#%%
def productAdjacency(molecA, molecB):
    iblA = index_by_label(molecA)
    iblB = index_by_label(molecB)
    compop = np.zeros(min(len(iblA), len(iblB)))
    for k in range(len(compop)):
        compop[k] += len(iblA[k])*len(iblB[k])

    # strategy:  compute the adjacency matrix of strict product graph
    # Then discard the connections to non-existent atoms
    
    # strict adjmat
    adjmatA = nx.adjacency_matrix(molecA)
    adjmatB = nx.adjacency_matrix(molecB)
    adjmat = sparse.kron(adjmatA.astype(int), adjmatB.astype(int))
    # Now make a list of product atoms
    keeplist = [] # strict product vertices to product graph index
    pgcd = []     # product graph index to (molecA, molecB) coordinates
    for k in range(len(iblA)):
        modA = len(list(molecB._node.keys()))
        for idxA in iblA[k]:
            if k >= len(iblB):
                break
            for idxB in iblB[k]:
                keeplist.append(int(idxA*modA+idxB))
                pgcd.append((idxA, idxB))
    if len(keeplist)!=int(compop.sum()):
        print(f" WARNING - keeplist {len(keeplist)}!= compop.sum() {int(compop.sum())}")
    # delete cols/rows from strict adjmat corresponding to illegal atoms
    # pgcd stores the prod graph vertices as 
    # pgcd[v] = (vA, vB)
    adjmat = adjmat.toarray()[:,keeplist][keeplist,:]
    adjmat = sparse.coo_matrix(adjmat)
    return adjmat, pgcd
adjmat = productAdjacency(train[51], train[51])


# small matrix indices for testing against manual calculation:
# 948, 1020, 1302, 1515, 1821, 1851, 1877, 1901,
    #   2473, 2501, 2518, 2555, 2634, 2635,


class GraphKernel():
    def __init__(self, order=10, kerneltype="order"):
        self.order=order
        self.beta=0.25
        self.type=kerneltype
        self.pi0 = "uniform"


    def kernel(self, molecA, molecB):
        outval = 0
        NomialMat,cds = productAdjacency(molecA, molecB)
        normalize = 10
        if self.pi0=="uniform":
            basevec = np.ones(NomialMat.shape[0])
        else:
            pass
        
        for i in range(self.order):
            if i>0:
                NomialMat = NomialMat.dot(NomialMat)/normalize
            if self.type=="order":
                wgt = int(i==self.order-1)

            elif self.type=="geometric":
                wgt = self.beta**(i+1)
            else:
                pass

            if np.sum(NomialMat<0)>0:
                print(f" WARNING - negative value in {i}-th degree prod adjmat")
                print(np.min(NomialMat[NomialMat<0]))

            outval += wgt*np.sum(basevec@NomialMat)
        return outval

if __name__=="__main__":
    Sixth = GraphKernel(order=6)
    idxA = np.random.randint(0,1000)
    idxB = np.random.randint(0,1000)
    idxA, idxB = 571, 50
    print(idxA, idxB)
    print(Sixth.kernel(train[idxA],train[idxA]))
    print(Sixth.kernel(train[idxB],train[idxB]))
    print(Sixth.kernel(train[idxA],train[idxB]))

    Geometric = GraphKernel(order=6, kerneltype="geometric")
    Geometric.kernel(train[100], train[100])

