def add_dict_counts(dict_a, dict_b):
    return {key: dict_a.get(key, 0) + dict_b.get(key, 0) for key in set(dict_a) | set(dict_b)}
