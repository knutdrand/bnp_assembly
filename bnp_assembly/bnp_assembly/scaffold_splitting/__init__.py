from dataclasses import dataclass
from collections import Counter
# from functools import cache, cached_property
import typing as tp

import numpy as np


@dataclass
class State:
    _max_length: int
    locations: tp.List[int]
    edges: tp.List[int]
    node_counts: tp.List[int]
    counter: Counter
    def update_state(self, location):
        locations.append(location)
        while locations[0] < location-self._max_length:
            locations.pop_left()
            edge_counts[0] -= 1
            if edge_counts[0] == 0:
                edge_counts.pop_left()
                edges.pop_left()
            for edge, count in zip(edges, counts):
                counter[edge] += 1


def count_possible_edge_pairs(locations: tp.List[int], edge_locations: tp.List[int], max_length: int):
    counter = [0 for edge in edge_locations]
    for edge_idx, edge_location in enumerate(edge_locations):
        for i, location in enumerate(locations):
            if location >= edge_location:
                break
            if location < edge_location-max_length:
                continue
            for location2 in locations[i+1:]:
                if location2 < edge_location:
                    continue
                if location2 > location+max_length:
                    break
                counter[edge_idx] += 1
    return counter


def count_edge_overlaps(locations_a, locations_b, edge_locations, max_length):
    locations_a = np.asarray(locations_a)
    locations_b = np.asarray(locations_b)
    edge_locations = np.asarray(edge_locations)
    start_array = np.zeros(len(edge_locations)+1)
    end_array = np.zeros(len(edge_locations)+1)
    distance = np.abs(locations_a-locations_b)
    first_location = np.minimum(locations_a, locations_b)
    second_location = np.maximum(locations_a, locations_b)
    mask = distance <= max_length
    first_location = first_location[mask]
    second_location = second_location[mask]
    start_indices = np.searchsorted(edge_locations, first_location, side='right')
    end_indices = np.searchsorted(edge_locations, second_location, side='right')
    start_array = np.bincount(start_indices, minlength=len(start_array))
    end_array = np.bincount(end_indices, minlength=len(end_array))
    start_count = np.cumsum(start_array)
    end_count = np.cumsum(end_array)
    return (start_count - end_count)[:-1]

def _count_possible_edge_pairs(locations, edge_locations, max_length):
    counter = Counter()
    window_start = 0
    windown_stop = 0
    edge_iter = iter(edge_locations)
    edge_buffer = []
    buffer_counts = []
    next_edge_position = next(edge_iter)
    for location in locations:
        if edge_buffer:
            pass

count_possible_edge_pairs([2, 3, 5, 7, 11], [2, 4, 8, 16], 4)





'''
class EdgeProbabilities:

    @cache
    def non_edge_rate(self, edge_i, edge_j):
        pass

    @cache
    def edge_rate(self, edge_i, edge_j):
        pass


    def _calculate_edge_rates(self, read_locations, edge_locations, window_size):
        window_start = 0
        windown_stop = 0
        edge_iter = iter(edge_locations)
        edge_buffer = [edge]
        next_edge_position = next(edge_iter)
        for location in locations:
            if edge_buffer:

        while True:


        read_iter = iter(read_locations)


        while True:



'''
