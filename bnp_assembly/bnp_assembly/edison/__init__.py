from ..agp import ScaffoldAlignments
from .scaffold_distance import distance

def get_scaffold_distance(true_agp: ScaffoldAlignments, estimated_agp: ScaffoldAlignments):
    distance(true_agp.to_dict(), estimated_agp.to_dict())
