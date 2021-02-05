import copy
import queue
import numpy as np
import pandas as pd

class BoundingBox:
    def __init__(self, p1=np.full(2, np.inf), p2=np.full(2, -np.inf)):
        self.extent = np.array([np.array(p1, dtype=np.float64),
                                np.array(p2, dtype=np.float64)])
    def __getitem__(self, idx):
        return self.extent[idx]
    def __repr__(self):
        return self.extent.__repr__()
    def __eq__(self, other):
        if isinstance(other, BoundingBox):
            return np.array_equal(self.extent, other.extent)
        return NotImplemented

    def sum_interval(self):
        return np.sum(np.diff(self.extent, axis=0))
    def contains(self, p):
        return np.all(np.logical_and(self[0] <= p, p <= self[1]))
    def union(self, bbox):
        p1 = np.min(np.array([self[0], bbox[0]]), axis=0)
        p2 = np.max(np.array([self[1], bbox[1]]), axis=0)
        return BoundingBox(p1, p2)

    def split(self, axis, cut):
        bb1 = copy.deepcopy(self)
        bb1[1][axis] = cut
        bb2 = copy.deepcopy(self)
        bb2[0][axis] = cut
        return (bb1, bb2)

def create_bounds(df, margin=0):
    extent = df[['x', 'y']].agg(['min', 'max'])
    p1 = extent.loc['min'].to_numpy() - margin
    p2 = extent.loc['max'].to_numpy() + margin
    return BoundingBox(p1, p2)

class MondrianNode:
    def __init__(self, bbox=BoundingBox(), tau=0, height=0):
        self.bbox = bbox
        self.tau = tau
        self.height = height
        self.axis = 0
        self.cut = 0
        self.leaf_idx = -1
        self.children = [None, None]

    def is_leaf(self):
        return all(child is None for child in self.children)

class MondrianTree:
    def __init__(self, l=0, data_window=BoundingBox(), seed=None):
        self.lamda = l
        self.bbox = data_window
        self.seed = seed
        self.root = None
        self.inner = []
        self.leaves = []
        self.init_tree()

    def init_tree(self):
        rng = np.random.default_rng(self.seed)
        bfs = queue.SimpleQueue()
        node = MondrianNode(self.bbox, 0, 0)
        self.root = node
        if self.lamda > 0:
            self.inner.append(node)
            bfs.put(node)
        else:
            self.leaves.append(node)

        while not bfs.empty():
            node = bfs.get()
            bbox = node.bbox
            node.axis = rng.integers(0, 1, endpoint=True)
            cut_interval = bbox[1][node.axis] - bbox[0][node.axis]
            node.cut = bbox[0][node.axis] + cut_interval * rng.uniform()
            height_children = node.height + 1
            bbox_children = bbox.split(node.axis, node.cut)

            # Create children nodes and compute their cost
            for idx, bb in enumerate(bbox_children):
                length = bb.sum_interval()
                tau_child = node.tau + rng.exponential(1 / length)
                node_child = MondrianNode(bb, tau_child, height_children)
                node.children[idx] = node_child

                # Check if child node can split further
                if tau_child < self.lamda:
                    self.inner.append(node_child)
                    bfs.put(node_child)
                # If node cost surpasses lambda, then it is a leaf node
                else:
                    self.leaves.append(node_child)

        for idx, leaf in enumerate(self.leaves):
            leaf.leaf_idx = idx

    def search_point(self, point):
        node = self.root
        if node.bbox.contains(point):
            while not node.is_leaf():
                node = node.children[0] if point[node.axis] < node.cut else node.children[1]
            return node
        return None

class MondrianForest:
    def __init__(self, forest_size=1, l=0, bbox=BoundingBox(), seed_seq=None):
        self.size = forest_size
        self.lamda = l
        self.bbox = bbox
        ss = np.random.SeedSequence(seed_seq)
        self.seeds = ss.spawn(self.size)
        self.trees = [MondrianTree(self.lamda, self.bbox, s) for s in self.seeds]
