import pytest
from pyESI.mondrian_tree import BoundingBox

def test_bbox_contains():
    bb = BoundingBox([0, 0], [3, 3])

    assert bb.contains([0.5, 2]) # Check inside point
    assert not bb.contains([4, 6]) # Check outside point
    assert bb.contains([0, 1]) # Check box vertices
    assert bb.contains([1, 3])
    assert bb.contains([1, 3]) # Check box borders
    assert bb.contains([0, 2])

def test_bbox_union():
    bb1 = BoundingBox([0, 0], [1, 1])
    bb2 = BoundingBox([2, 2], [3, 3])
    union1 = bb1.union(bb2) # Disjoint bounding boxes
    assert (union1[0] == bb1[0]).all
    assert (union1[1] == bb2[1]).all

    bb3 = BoundingBox([0.5, 0.5], [3, 3])
    union2 = bb1.union(bb3)
    assert (union2[0] == bb1[0]).all
    assert (union2[1] == bb3[1]).all

    assert BoundingBox().union(bb1) == bb1

def test_bbox_split():
    bb = BoundingBox([0, 0], [1, 1])
    axes = [0, 1]
    for axis in axes: # Try splitting over X and Y axes
        bb1, bb2 = bb.split(axis, 0.5)
        assert (bb1[0] == bb[0]).all
        assert (bb2[1] == bb[1]).all
        assert (bb1[1][axis] == 0.5)
        assert (bb2[0][axis] == 0.5)

def test_bbox_length():
    bb1 = BoundingBox([0, 0], [1, 1])
    assert bb1.sum_interval() == 2.
    bb2 = BoundingBox([0, 1], [3, 4])
    assert bb2.sum_interval() == 6.
