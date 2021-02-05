import pytest
import os
import numpy as np
import pandas as pd
import random
from pyESI.mondrian_tree import create_bounds
from pyESI.ensemble import EnsembleIDW, Partition

@pytest.fixture(scope="function")
def initialize(request):
    os.chdir(request.fspath.dirname)

def test_partition_sizes(initialize):
    coordinates = [[random.randrange(1, 200, 2), random.randrange(1, 300, 2),
                    random.uniform(0, 5)] for i in range(400)]
    coordinates = np.array(coordinates)
    samples = pd.DataFrame({'x': coordinates[:, 0], 'y': coordinates[:, 1],
                            'grade': coordinates[:, 2]})
    sample_count = len(samples.index)

    origin = create_bounds(samples)
    esi = EnsembleIDW(10, 0.5, origin, samples)
    for partition in esi.ensemble:
        sizes = [len(leaf) for leaf in partition.data]
        assert sum(sizes) == sample_count
