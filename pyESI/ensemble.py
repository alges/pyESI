import numpy as np
from pyESI.mondrian_tree import MondrianForest, create_bounds

def idw_interpolation(point, data):
    dists = np.linalg.norm(data[['x', 'y']].values - point, axis=1)
    weights = 1. / (1. + dists)
    w_norm = weights / np.sum(weights)
    return np.sum(w_norm * data['grade'])

class Partition:
    def __init__(self, tree, samples):
        bboxes = [leaf.bbox for leaf in tree.leaves]
        query = 'x >= @b[0][0] and x <= @b[1][0] and y >= @b[0][1] and y <= @b[1][1]'
        self.data = [samples.query(query) for b in bboxes]
        self.leaf_indices = [tree.search_point(point).leaf_idx for point in samples[['x','y']].values]

class EnsembleIDW:
    def __init__(self, size, alpha, bbox, samples):
        self.size = size
        self.alpha = alpha
        self.bbox = bbox
        self.samples = samples
        length = self.bbox.sum_interval()
        self.lamda = 1 / (length - self.alpha * length)
        self.forest = MondrianForest(self.size, self.lamda, self.bbox)
        self.ensemble = [Partition(tree, samples) for tree in self.forest.trees]

    def predict(self, points):
        predictions = []
        for point in points[['x','y']].values:
            values = []
            for tree_idx, tree in enumerate(self.forest.trees):
                leaf = tree.search_point(point)
                if leaf is not None:
                    neighbors = self.ensemble[tree_idx].data[leaf.leaf_idx]
                    if not neighbors.empty:
                        pred = idw_interpolation(point, neighbors)
                        values.append(pred)
            predictions.append(np.array(values) if values else np.array([-99.]))
        return Reduction(predictions)

    def cross_validation(self):
        predictions = []
        for point_idx in range(len(self.samples.index)):
            values = []
            for tree_idx, tree in enumerate(self.forest.trees):
                leaf_idx = self.ensemble[tree_idx].leaf_indices[point_idx]
                neighbors = self.ensemble[tree_idx].data[leaf_idx].drop(index=point_idx)
                point = self.samples.loc[point_idx][['x','y']].values
                pred = idw_interpolation(point, neighbors)
                values.append(pred)
            predictions.append(np.array(values) if values else np.array([-99.]))
        return Reduction(predictions)

class Reduction:
    def __init__(self, values):
        self.estimates = np.array([np.mean(p) for p in values])
        self.variances = np.array([np.var(p) for p in values])
