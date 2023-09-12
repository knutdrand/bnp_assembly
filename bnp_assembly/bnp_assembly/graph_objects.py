from dataclasses import dataclass
import typing as tp
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import poisson



@dataclass
class NodeSide:
    node_id: int
    side: str

    @property
    def contig_id(self):
        return self.node_id

    @property
    def numeric_index(self):
        return int(self.node_id*2 + (self.side == 'r'))

    @classmethod
    def from_numeric_index(cls, idx: int):
        return cls(idx//2, 'r' if idx % 2 == 1 else 'l')

    def other_side(self):
        return self.__class__(self.node_id, 'r' if self.side == 'l' else 'l')

    def to_directed_node(self):
        from bnp_assembly.contig_graph import DirectedNode
        return DirectedNode(self.node_id, '+' if self.side == 'r' else '-')

    def __hash__(self):
        if isinstance(self.node_id, str):
            return hash((self.node_id, self.side))
        return self.numeric_index

    def __repr__(self):
        return f'N({self.node_id}, {self.side})'

    @classmethod
    def from_string(cls, s):
        print(s.strip()[2:-1])
        node_id, side = s.strip()[2:-1].split(', ')
        return cls(int(node_id), side)

    def __eq__(self, other: 'NodeSide'):
        return self.node_id == other.node_id and self.side == other.side


@dataclass
class Edge:
    from_node_side: NodeSide
    to_node_side: NodeSide

    @property
    def numeric_index(self):
        return (self.from_node_side.numeric_index,
                self.to_node_side.numeric_index)

    def __hash__(self):
        if any(isinstance(v.node_id, str) for v in (self.to_node_side, self.from_node_side)):
            return hash((self.from_node_side, self.to_node_side))
        return hash(self.numeric_index)

    @classmethod
    def from_numeric_index(cls, idx):
        return cls(*(NodeSide.from_numeric_index(i) for i in idx))

    def reverse(self):
        return self.__class__(self.to_node_side,
                              self.from_node_side)

    def __repr__(self):
        return f'E({self.from_node_side}--{self.to_node_side})'

    @classmethod
    def from_string(cls, s):
        t = s.strip()[2:-1].split('--')
        return cls(*(NodeSide.from_string(v) for v in t))

