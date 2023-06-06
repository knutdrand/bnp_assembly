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
    def numeric_index(self):
        return int(self.node_id*2 + (self.side == 'r'))

    @classmethod
    def from_numeric_index(cls, idx: int):
        return cls(idx//2, 'r' if idx % 2 == 1 else 'l')

    def other_side(self):
        return self.__class__(self.node_id, 'r' if self.side == 'l' else 'l')

    def __hash__(self):
        if isinstance(self.node_id, str):
            return hash((self.node_id, self.side))
        return self.numeric_index

    def __repr__(self):
        return f'N({self.node_id}, {self.side})'

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
        return f'E({self.from_node_side}, {self.to_node_side})'
