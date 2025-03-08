from itertools import combinations, permutations
import numpy as np
import numpy.typing as npt
from typing import Any, Protocol
from random import shuffle


class CoalitionGenerator(Protocol):
    def __call__(self, input: list[Any], maxN: int) -> list[tuple[Any, ...]]:
        ...


def defaultCoalitionGenerator(
        input: list[Any], maxN: int) -> list[tuple[Any, ...]]:
    """Get all permutations of the input up to length `maxN`, equivalent to `itertools.combinations`"""

    return [x for i in range(1, maxN+1) for x in combinations(input, i)]


def createNDistinctCombinations(
    input: list[Any], maxN: int
) -> list[tuple[Any, ...]]:
    """
    Attempts to take up to n random combinations of the input array. 
    Optionally `permuteRange` can be specified in order to only permute a part of the input, keeping other parts untouched.

    To prevent infinite loops, permutation generation will stop after `15` successive failure to generate a unique permutation.

    Args:
        input: The input preference list
        maxN: The maximal number of permutations that is desired. This function has flexibility to return fewer if inputs are too small, or too inefficient.
        permuteRange: This range indicates the section of the array where permutation is desired. Leave blank to indicate full array

    Returns:
        A list of tuples, with each tuple being a unique permutation.
    """
    combo = list[tuple]()

    uniqueMiddles = set()
    countAttempts = 0
    successiveFailedAttempts = 0

    while len(uniqueMiddles) < maxN and successiveFailedAttempts < 15:
        countAttempts += 1
        shuffle(input)
        outputTuple = tuple(input)
        if (outputTuple in uniqueMiddles):
            successiveFailedAttempts += 1
            continue
        uniqueMiddles.add(outputTuple)
        successiveFailedAttempts = 0

    for i in uniqueMiddles:
        combo.append(i)

    return combo
