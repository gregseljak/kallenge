#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import networkx as nx
# Remark: no positive results above len==55
train = pkl.load(open("./training_data.pkl", "rb"))
trainy = pkl.load(open("./training_labels.pkl", "rb"))
nodel = np.zeros(len(train))
for i in range(len(nodel)):
    nodel[i]=len(train[i].nodes)


def display_molec(idx, _ax=None):
    if _ax is None:
        fig, ax = plt.subplots()
    else:
        ax = _ax
    molec = train[idx]
    atomtype = {}
    bondtype = {}
    for i in range(len(molec.nodes)):
        atomtype[i] = str(molec.nodes[i]["labels"][0])
        for ii in list(molec._adj[i].keys()):
            a, b = min(i,ii), max(i,ii)
            bondtype[(a,b)] = str(molec._adj[i][ii]["labels"][0])

    pos = nx.kamada_kawai_layout(molec)
    nx.draw(molec,with_labels=True, labels=atomtype, pos=pos, ax=ax)
    nx.draw_networkx_edge_labels(molec, pos=pos, edge_labels=bondtype, ax=ax)
    ax.set_title(f"idx {idx}, classement {trainy[idx]}")

if __name__=="__main__":
    pos_idxs = np.where(trainy==1)[0]
    display_molec(pos_idxs[16])

