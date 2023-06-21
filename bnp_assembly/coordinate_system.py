from bnp_assembly.graph_objects import NodeSide


class CoordinateSystem:
    def __init__(self, contig_dict, edges):
        self._contig_dict = contig_dict
        self._edges = edges
        self._node_sides = {edge.from_node_side for edge in edges} | {edge.to_node_side for edge in edges}
        self._node_side_dict = {node_side.node_id: node_side for node_side in self._node_sides}
        self._edge_dict = {(edge.from_node_side.node_id, edge.to_node_side.contig_id): edge for edge in edges}

    def location_coordinates(self, location):
        return self.node_side_coordinates(self._node_side_dict[int(location.contig_id)], location.offset)

    def location_pair_coordinates(self, location_pair):
        a, b = location_pair.location_a, location_pair.location_b
        if (int(b.contig_id), int(a.contig_id)) in self._edge_dict:
            a, b = b, a
        edge = self._edge_dict[(int(a.contig_id), int(b.contig_id))]
        return edge, (self.node_side_coordinates(edge.from_node_side, a.offset), self.node_side_coordinates(edge.to_node_side, b.offset))
        # self._node_side_dict[int(a.contig_id)], a.offset), self.node_side_coordinates(self._node_side_dict[int(b.contig_id)], b.offset))

    def __contains__(self, location_pair):
        a, b = int(location_pair.location_a.contig_id), int(location_pair.location_b.contig_id)
        return ((a, b) in self._edge_dict) or ((b, a) in self._edge_dict)

    def node_side_coordinates(self, node_side: NodeSide, offset):
        if node_side.side == 'l':
            r = offset
        else:
            assert node_side.side == 'r', node_side
            r = self._contig_dict[int(node_side.contig_id)] - offset-1
        assert r >= 0, (node_side, offset)
        return r
