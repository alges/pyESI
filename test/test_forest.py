import pytest
from pyESI.mondrian_tree import MondrianForest, BoundingBox

def test_leaf_boxes():
    origin = BoundingBox([1, 1], [399, 599])
    length = origin.sum_interval()
    alpha = 0.5
    lamda = 1 / (length - alpha * length)
    forest = MondrianForest(10, lamda, origin)
    for tree in forest.trees:
        bbox = BoundingBox()
        for leaf in tree.leaves:
            bbox = bbox.union(leaf.bbox)
        assert bbox == origin
