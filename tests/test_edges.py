from bnp_assembly.hic_distance_matrix import Edge, NodeSide


def test_numeric_index_nodeside():
    for i in range(10):
        assert NodeSide.from_numeric_index(i).numeric_index == i


def test_numeric_index_edge():
    for i in range(10):
        for j in range(10):
            assert Edge.from_numeric_index((i, j)).numeric_index == (i, j)

def test_reverse():
    edge = Edge(NodeSide(1, 'l'),
                NodeSide(0, 'r'))
    reverse_edge = Edge(NodeSide(0, 'r'),
                        NodeSide(1, 'l'))
    print(edge.reverse())
    assert edge.reverse() == reverse_edge
