def drop_element(vec_dict, key):
    new_dict = dict(vec_dict)
    val = new_dict[key]
    del new_dict[key]
    return val, new_dict


# def find_nearest(value, vec_dict):
#     min_key, min_value = min(vec_dict.items(), key=lambda tup: abs(tup[1] - target))
    # return min_value


def sort_dict(vec_dict, init_idx, length):
    sorted_dict = {}

    cur_val, vec_dict = drop_element(vec_dict, init_idx)
    sorted_dict[init_idx] = cur_val

    # nxt_idx = find_nearest(cur_val, vec_dict)

#     for i in range(road_length):
#         nxt_idx = find_nearest(arr, cur_val)
#         cur_val, arr = drop_element(arr, nxt_idx)
#         sorted_vectors.append(cur_val)

#     return sorted_vectors
