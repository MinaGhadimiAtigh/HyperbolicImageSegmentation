import logging
import math
import queue

import networkx as nx
import numpy as np

from hesp.hierarchy.hierarchy_helpers import json2rels, hierarchy_pos
from hesp.hierarchy.node import Node
from hesp.visualize.visualize_helpers import colour_nodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tree(object):
    """ Tree object containing the target classes and their hierarchical relationships.
    Args:
        i2c: dict linking label indices to concept names. 
        json: json representation of hierarchical relationships. Empty dict assumes no hierarchical relationships.
    
    """

    def __init__(self, i2c, json={}):
        self.i2c = i2c
        self.K = max([i for i, c in self.i2c.items()]) + 1  # num classes = max int of class
        self.json = json
        if len(self.json) == 0:  # no json provided, assume flat
            self.json = {"root": {c: {} for _, c in i2c.items()}}

        self.root = list(self.json.keys())[0]
        self.target_classes = np.array([i for i, c in self.i2c.items()])
        self.c2i = {c: i for i, c in self.i2c.items()}
        self.nodes = self.init_nodes()
        self.train_classes = list(self.i2n.keys())
        self.M = self.get_M()  # len(self.nodes) - 1  # excl root
        self.init_matrices()
        self.nodes = colour_nodes(self.nodes, self.root)
        self.init_graph()
        self.init_metric_families()

    def get_sibling_nodes(self, node_name):
        return [self.nodes[s] for s in self.nodes[node_name].siblings]

    def get_parent_node(self, node_name):
        return self.nodes[self.nodes[node_name].parent]

    def get_M(self, ):
        ancestor_nodes = [n for n in self.nodes if n not in self.c2i]
        return self.K + len(ancestor_nodes) - 1  # excl root

    def get_by_idx(self, idx):
        return self.nodes[self.i2n[idx]]

    def init_nodes(self, ):
        """
        Initialize node objects and node 2 index dicts.
        """
        idx_counter = self.K
        self.i2n = self.i2c.copy()
        q = queue.Queue()
        nodes = {}

        root_node = Node(
            name=self.root,
            parent=None,
            children=list(self.json[self.root].keys()),
            ancestors=[],
            siblings=[],
            depth=0,
            sub_hierarchy=self.json[self.root],
            idx=-1,
        )
        nodes[self.root] = root_node
        q.put(root_node)
        while not q.empty():
            parent = q.get()
            for c in parent.children:
                if c in self.c2i:
                    idx = self.c2i[c]
                else:
                    idx = idx_counter
                    idx_counter += 1
                child_node = Node(
                    name=c,
                    parent=parent.name,
                    children=list(parent.sub_hierarchy[c].keys()),
                    ancestors=parent.ancestors + [parent.name],
                    siblings=parent.children,
                    depth=parent.depth + 1,
                    sub_hierarchy=parent.sub_hierarchy[c],
                    idx=idx,
                )
                if idx not in self.i2n:
                    self.i2n[idx] = c
                nodes[c] = child_node
                q.put(child_node)
        self.n2i = {n: i for i, n in self.i2n.items()}
        return nodes

    def init_matrices(self, ):
        """ Build 'hmat', e.g. connection matrix of the hierarchy and
        'sibmat', e.g. matrix containing sibling connections
        """
        self.hmat = np.zeros((self.M, self.M), dtype=np.float32)
        self.sibmat = np.zeros((self.M, self.M), dtype=np.float32)
        for i in range(self.M):  # excl. root
            if i in self.i2n:
                concept = self.i2n[i]
                csib_idx = [self.n2i[s] for s in self.nodes[concept].siblings]
                self.sibmat[i] = np.array([i in csib_idx for i in range(self.M)]).astype(np.float32)
                chierch_idx = [i] + [
                    self.n2i[a] for a in self.nodes[concept].ancestors if a != self.root
                ]
                self.hmat[i] = np.array([i in chierch_idx for i in range(self.M)]).astype(np.float32)

    def init_graph(self, ):
        """ Initializes networkx graph, used for visualization. """
        rels = json2rels(self.json)
        self.G = nx.Graph()
        self.G.add_edges_from(rels)
        pos = hierarchy_pos(self.G, self.root, width=2 * math.pi)
        self.pos = {
            u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()
        }

    @property
    def levels(self, ):
        """ Returns nodes in order of depth """
        max_depth = np.max([n.depth for _, n in self.nodes.items()])
        stages = []
        for i in range(max_depth):
            stage = [n for _, n in self.nodes.items() if n.depth == i + 1]
            stages.append(stage)
        return stages

    def is_hyponym_of(self, key, target):
        if self.nodes[key].parent is None:
            return False
        if self.nodes[key].parent == target:
            return True
        else:
            return self.is_hyponym_of(self.nodes[key].parent, target)

    def metric_family(self, concept):
        node = self.nodes[concept]
        siblings = [i for i in self.target_classes if self.is_hyponym_of(self.i2c[i], node.parent)]
        cousins = [i for i in self.target_classes if self.is_hyponym_of(self.i2c[i], self.nodes[node.parent].parent)]
        return siblings, cousins

    def init_metric_families(self, ):
        for i in self.target_classes:
            name = self.i2c[i]
            node = self.nodes[name]

            metric_siblings, metric_cousins = self.metric_family(name)
            if node.parent != 'root':
                node.metric_siblings = metric_siblings
            else:
                # parent is root, no hierarchical relaxation as it would include all nodes
                node.metric_siblings = [i]
                node.metric_cousins = [i]
                continue

            if self.nodes[node.parent].parent != 'root':
                # we know the parent is not root if we are here
                node.metric_cousins = metric_cousins
            else:
                node.metric_cousins = metric_siblings
