
def get_stats(ids: list, counts=None):
    """
    Get the frequency of each pair of ids in a list of ids.

    Args:
        ids (list): List of ids.
        counts (dict, optional): Dictionary of counts. Defaults to None.

    Returns:
        dict: Dictionary of counts.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: list, pair, idx: int):
    """
    Merge a pair of ids in a list of ids.

    Args:
        ids (list): List of ids.
        pair (): Pair of ids to merge.
        idx (int): Index to replace the pair with.

    Returns:
        list: New list of ids.
    """

    newids = []  # new list of ids
    i = 0
    while i < len(ids):
        #  if not at the very last position AND the pair matches, replace it
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids