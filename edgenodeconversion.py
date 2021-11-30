import numpy as np

def nodes_to_edges(n1,n2,df):
    N1 = np.array(df["End1"])
    N2 = np.array(df["End2"])

    ind1A = np.argwhere(N1 == n1)
    ind2A = np.argwhere(N2 == n2)
    
    ind1B = np.argwhere(N1 == n2)
    ind2B = np.argwhere(N2 == n1)
    
    edgeA = np.intersect1d(ind1A,ind2A)
    edgeB = np.intersect1d(ind1B,ind2B)
    
    if edgeA.size == 0:
        return np.squeeze(edgeB)
    elif edgeB.size == 0:
        return np.squeeze(edgeA)
    else:
        print("Inside nodes_to_edges")
        print("Nodes not connected")
        return -1
    
def edges_to_nodes(e,df):
    return np.squeeze(np.array(df["End1"][e])), np.squeeze(np.array(df["End2"][e]))
