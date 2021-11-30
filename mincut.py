import networkx as nx
import numpy as np

def MinimumCutPhase(G, a):
    '''
    Maximum adjacency search.
    Inputs:
        G: networkx Graph object
        a: starting node

    Outputs:
        G: Graph with merged nodes
        w: weight of cut-of-the-phase
        cotp: cut-of-the-phase node
    '''
    A = [a]

    while set(A) != set(G.nodes()):
        v = find_mc_vertex(G, A)    # find most tightly connected vertex
        A.append(v)                 # add to A the most tightly connected vertex

    cotp = A[-1]    # store cut-of-the-phase

    # compute weight of cut-of-the-phase
    w = 0
    for node in G.neighbors(cotp):
        # print('add weight', G[cotp][node]["weight"])
        w += G[cotp][node]["weight"]

    G = merge_vertices(G, A[-2], A[-1]) # merge two vertices added last
    return G, w, cotp

def MinimumCut(G, a):
    '''
    Finds minimum cut.
    Inputs:
        G: networkx Graph object
        a: starting node

    Outputs:
        min_cut: set of nodes in minimum cut
        min_w:   weight of minimum cut
    '''
    min_cut = set([])
    min_w = np.inf

    Gcp = G.copy()
    while Gcp.number_of_nodes() > 1:
        Gcp, w, cotp = MinimumCutPhase(Gcp, a)
        if w < min_w:
            min_w = w
            min_cut.add(cotp)
    return min_cut, min_w

def merge_vertices(G, u, v):
    '''
    Merge vertices u and v in the graph G.
    Inputs:
        G: networkx Graph object
        u: merged node that remains in graph
        v: merged node that is removed from graph
    Outputs:
        G: Graph with merged nodes
    '''

    for node in G.neighbors(v):
        if G.has_edge(u,node):  # adds weights of edges to shared node
            G[u][node]["weight"] += G[v][node]["weight"]
        elif node != u:         # adds edge
            G.add_edge(u, node, weight=G[v][node]["weight"])

    G.remove_node(v)
    return G

def find_mc_vertex(G, A):
    '''
    Finds most tightly connected vertex in G\A to A.
    Inputs:
        G: networkx Graph object
        A: Subset of G.nodes

    Outputs:
        mc_v: most tightly connected vertex
    '''
    max_weight = -1
    mc_v = None

    for v in G:
        w = 0
        if v not in A:
            for u in A:
                if G.has_edge(u,v):
                    w += G[u][v]["weight"]
            if w > max_weight:
                max_weight = w
                mc_v = v
    return mc_v

def main():
    G = nx.Graph()
    G.add_edge('x', 'a', weight=3)
    G.add_edge('x', 'b', weight=1)
    G.add_edge('a', 'c', weight=3)
    G.add_edge('b', 'c', weight=5)
    G.add_edge('b', 'd', weight=4)
    G.add_edge('d', 'e', weight=2)
    G.add_edge('c', 'y', weight=2)
    G.add_edge('e', 'y', weight=3)

    print('G.nodes =', G.nodes)
    min_cut, min_w = MinimumCut(G, 'a')
    print('min_cut =', min_cut)
    print('min_w =', min_w)
    print('G.nodes =', G.nodes)
    print(G.nodes['x'])

    # cut_value, partition = nx.stoer_wagner(G)
    # print('cut_value = ', cut_value)
    # print('partition', partition)

if __name__ == '__main__':
    main()
