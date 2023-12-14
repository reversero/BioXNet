import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import re
import networkx as nx
import pandas as pd
from os.path import join
from gmt_reader import GMT

def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G

def complete_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    # distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        # distance from the root nodes to the terminal (final) nodes
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            # suppose a terminal node named 'node_t', if its distance from the root node is smaller 
            # than n_levels, we will create a copy named 'node_t_copy' and add an edge 'node_t' -> 'node_t_copy'
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph

def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))

    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        # remove the nodes whose distance is smaller than the given distance
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)

def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n) # remove the _copy from its original names
            next = net.successors(n)
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
        layers.append(dict)
    return layers

class Reactome():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pathway_names = self.load_names()
        self.hierarchy = self.load_hierarchy()
        self.pathway_genes = self.load_genes()

    def load_names(self):
        df = pd.read_csv(join(self.data_dir, 'ReactomePathways.txt'), sep='\t')
        df.columns = ['reactome_id', 'pathway_name', 'species']
        return df

    def load_genes(self):
        gmt = GMT()
        df = gmt.load_data(join(self.data_dir, 'ReactomePathways.gmt'), 
                           pathway_col=1, genes_col=3)
        '''The columns of df are group (pathway) and gene'''
        return df

    def load_hierarchy(self):
        df = pd.read_csv(join(self.data_dir, 'ReactomePathwaysRelation.txt'), sep='\t')
        # child -> parent: child participates the event of patent
        df.columns = ['child', 'parent']
        return df

class ReactomeNetwork():
    def __init__(self, data_dir):
        self.reactome = Reactome(data_dir)  # low level access to reactome pathways and genes
        self.netx = self.get_reactome_networkx()

    def get_terminals(self):
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes

    def get_roots(self):
        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    # get a DiGraph representation of the Reactome hierarchy
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            # already created the network
            return self.netx
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
        net.name = 'reactome'

        # add root node
        roots = [n for n, d in net.in_degree() if d == 0] # for the nodes whose in-degree is 0
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):
        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')
        return G

    def get_completed_network(self, n_levels):
        G = complete_network(self.netx, n_leveles=n_levels)
        return G

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_leveles=n_levels)
        return G

    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            '''
                net (DiGraph): root -> pathway, no genes
                layers (list): each element is a dict, denote one layer, 
                               whose key is the node name and the values are its successors
            '''
            print('get layers from root to leaf')
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        # get the last layer (genes level)
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]  # set of terminal pathways
        # we need to find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes

        # There are no genes in previous layers. Add the genes that directly link to the pathways
        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            genes = genes_df[genes_df['group'] == pathway_name]['gene'].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes
        layers.append(dict)

        return layers