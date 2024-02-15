import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import re
import networkx as nx
import pandas as pd
from os.path import join
from gmt_reader import GMT


'''
    向图 G 中添加从节点 node 开始的多级边（参数 n_levels 指定要添加的边的级别数量）：
        - 创建一个空列表 edges，用于存储要添加到图中的边
        - 使用循环迭代 n_levels 次，依次创建从 node 到 node_copy1、node_copy2 等的边
        - 每次迭代，根据当前级别 l 构建目标节点 target，并创建一条从当前节点 source 到目标节点 target 的边，并将其添加到 edges 列表中
        - 将 target 设置为下一次迭代的 source，以便在下一级别中使用
        - 使用 add_edges_from 方法将 edges 列表中的所有边添加到图 G 中
        - 返回更新后的图 G
'''
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


'''
    完善一个给定的网络图 G，以确保从根节点到终端节点的距离至少为 n_levels：
        - 使用 nx.ego_graph(G, 'root', radius=n_leveles) 从图 G 中获取以根节点 'root' 为中心、半径为 n_leveles 的子图 sub_graph
        - 找到子图中的所有终端节点（即出度为0的节点），存储在列表 terminal_nodes 中
        - 对于每个终端节点 node:
            - 计算从根节点到终端节点的距离 distance，使用 nx.shortest_path(sub_graph, source='root', target=node) 计算最短路径的长度
            - 如果距离小于或等于 n_levels，则需要补充边，以确保距离达到 n_levels
            - 计算需要添加的边的数量 diff，即 n_levels - distance + 1
            - 调用 add_edges 函数将 diff 数量的边从终端节点添加到其相应的副本节点，以确保距离达到 n_levels
        - 返回更新后的子图 sub_graph
'''
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


'''
    获取在给定距离内的所有节点（NetworkX 中的 ego_graph 函数来获取指定节点周围的子图，并根据给定的距离过滤节点）：
        - net 是一个 NetworkX 图对象
        - distance 是一个整数，表示要获取节点的距离
        - 函数首先使用 ego_graph 函数获取以根节点（这里是 'root'）为中心、指定半径为 distance 的子图
        - 如果 distance 大于等于 1，函数会删除那些距离小于给定距离的节点。这是通过计算距离为 distance - 1 的子图并将其从节点集合中删除来完成的
        - 函数返回一个列表，其中包含在给定距离内的所有节点
'''
def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))

    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        # remove the nodes whose distance is smaller than the given distance
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)


'''
    获取网络图中与给定节点在特定距离内的所有节点：
        - 使用 nx.ego_graph(net, 'root', radius=distance) 获取以根节点 'root' 为中心、半径为 distance 的子图中的所有节点，并将其存储在集合 nodes 中
        - 如果指定的距离 distance 大于等于1，则需要移除那些距离小于给定距离的节点，以确保返回的节点是在指定距离上的节点
        - 返回存储节点的列表 nodes
'''
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


'''
    加载 Reactome 数据库的路径信息、层次结构以及与路径相关的基因信息：
        - load_names 方法用于加载路径的名称信息，从文件 ReactomePathways.txt 中读取，返回一个包含反应通路ID、通路名称和物种信息的 DataFrame
        - load_genes 方法用于加载与路径相关的基因信息，从文件 ReactomePathways.gmt 中读取，返回一个包含反应通路与基因之间关系的 DataFrame，其中每一行包含一个反应通路与一个基因之间的关联
        - load_hierarchy 方法用于加载路径之间的层次结构信息，从文件 ReactomePathwaysRelation.txt 中读取，返回一个包含子路径与父路径之间关系的 DataFrame，其中每一行表示子路径参与了父路径的事件
'''
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
        # child -> parent: child participates the event of parent
        df.columns = ['child', 'parent']
        return df


'''
    处理 Reactome 数据库中的网络信息，包括路径之间的关系、层次结构等
'''
class ReactomeNetwork():
    def __init__(self, data_dir):
        self.reactome = Reactome(data_dir)  # low level access to reactome pathways and genes
        self.netx = self.get_reactome_networkx()

    '''
        获取网络中的末端节点，即出度为0的节点（终点）
    '''
    def get_terminals(self):
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes


    '''
        获取网络中的根节点，即距离根节点为1的节点（起始节点）
    '''
    def get_roots(self):
        roots = get_nodes_at_level(self.netx, distance=1)
        return roots


    '''
       获取 Reactome 数据库的网络表示：
            - 如果已经创建了网络，就直接返回已经创建的网络对象
            - 否则，从 Reactome 数据库的层次结构中获取包含人类路径信息的子层次结构
                - 从 Reactome 数据库中加载层次结构数据
                - 从加载的数据中筛选出包含人类路径信息的子层次结构。这些信息通常以人类（Homo sapiens）基因的标识符开头（如"HSA"）
            - 使用 NetworkX 创建有向图
            - 添加根节点到图中
                - 通过检查每个节点的入度（in-degree），找到没有父节点的节点，即根节点的直接子节点
                - 创建一个名为 "root" 的虚拟节点，表示图的根节点
                - 将根节点与找到的直接子节点之间建立边
            - 返回创建的网络对象
    '''
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


    '''
       获取网络的信息，包括节点数量、边数量等（nx.info 被 networkx 3.x 移除）
    '''
    def info(self):
        return nx.info(self.netx)

    # def print_network_info(self):
    #     print(f'Number of nodes: {nx.number_of_nodes(self.netx)}')
    #     print(f'Number of edges: {nx.number_of_edges(self.netx)}')
    #     print(f'Average node degree: {nx.number_of_edges(self.netx) / nx.number_of_nodes(self.netx):.2f}')
    #     print(f'Is directed: {nx.is_directed(self.netx)}')

    '''
       获取网络的树形结构，通过对网络进行广度优先搜索（bfs）来构建树形结构
    '''
    def get_tree(self):
        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')
        return G


    '''
       获取完整的网络，即在网络中添加了多余的节点和边以达到指定的层次数
    '''
    def get_completed_network(self, n_levels):
        G = complete_network(self.netx, n_leveles=n_levels)
        return G


    '''
       获取完整的树形结构，即在树形结构中添加了多余的节点和边以达到指定的层次数
    '''
    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_leveles=n_levels)
        return G


    '''
       获取网络的各个层次：
        - 如果direction为 'root_to_leaf'，则从根节点开始获取网络的层次结构，并使用函数 get_layers_from_net 提取每一层的信息。
            返回的 layers 是一个列表，其中每个元素是一个字典，表示一个层次，字典的键是节点名称，值是该节点的后继节点集合
        - 如果direction不是 'root_to_leaf'，则默认从根节点到叶节点获取完整的网络，然后从中提取最后 n_levels 层的信息。这是因为整个网络中的路径都是从根节点开始到叶节点结束的
        - 获取最后一层（基因级别）的信息：
            - 查找网络中的终端节点，即出度为 0 的节点，这些节点代表了最终的途径
            - 根据这些途径的名称从 Reactome 数据库中获取与之关联的基因
            - 将这些基因添加到字典中，并将该字典添加到 layers 列表中
    '''
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