import math
from typing import Dict, Optional, Callable


def droop_quota(tot_votes: int, tot_seats: int) -> int:
    """
    Computes droop quota for given number of valid votes and available seats
    :param tot_votes: Total valid votes
    :param tot_seats: Total available seats
    :return: Droop quota
    """
    return math.floor(tot_votes/(tot_seats + 1) + 1)


def dict_argmax(
        d: Dict,
        key: Optional[Callable] = None
) -> int:
    """
    Gets argmax of supplied dictionary
    :param d: dictionary to get argmax off
    :param key: a function to specify the argmax criteria (similar to key argument of Python inbuilt 'sort')
    :return: index of max value
    """
    if key is None:
        key = lambda x: d

    max_val = None
    max_key = None
    for dict_key, dict_val in d.items():
        if max_val is None or key(dict_val) > key(max_val):
            max_val = dict_val
            max_key = dict_key

    return max_key
