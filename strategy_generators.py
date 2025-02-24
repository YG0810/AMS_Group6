from itertools import permutations
import numpy as np
from typing import Any, Protocol
from random import shuffle

class StrategyGenerator(Protocol):
    def __call__(self, input: np.char.chararray, maxN: int, permuteRange: range | None = None) -> list[tuple[str, ...]]:
        ...

def defaultStrategyGenerator(
        input: np.char.chararray, maxN: int, permuteRange: range | None = None) -> list[tuple[str, ...]]:
    """Get all permutations of the input, equivalent to `itertools.permutations`"""

    return list(permutations(input))


def createNDistinctPermutations(
    input: np.char.chararray, maxN: int, permuteRange: range | None = None
) -> list[tuple[str, ...]]:
    """
    Attempts to take up to n random permutations of the input array. 
    Optionally `permuteRange` can be specified in order to only permute a part of the input, keeping other parts untouched.

    To prevent infinite loops, permutation generation will stop after `15` successive failure to generate a unique permutation.

    Args:
        input: The input preference list
        maxN: The maximal number of permutations that is desired. This function has flexibility to return fewer if inputs are too small, or too inefficient.
        permuteRange: This range indicates the section of the array where permutation is desired. Leave blank to indicate full array

    Returns:
        A list of tuples, with each tuple being a unique permutation.
    """
    permu = list[tuple]()

    if permuteRange is None or permuteRange.stop > len(input):
        permuteRange = range(len(input))

    if len(permuteRange) == 0:
        return permu

    left: list[Any] = input[0: permuteRange.start].tolist()
    middle: list[Any] = input[permuteRange.start: permuteRange.stop].tolist()
    right: list[Any] = input[permuteRange.stop: len(input)].tolist()

    uniqueMiddles = set()
    countAttempts = 0
    successiveFailedAttempts = 0

    while len(uniqueMiddles) < maxN and successiveFailedAttempts < 15:
        countAttempts += 1
        shuffle(middle)
        outputTuple = tuple(middle)
        if (outputTuple in uniqueMiddles):
            successiveFailedAttempts += 1
            continue
        uniqueMiddles.add(outputTuple)
        successiveFailedAttempts = 0

    for i in uniqueMiddles:
        permu.append(tuple(np.char.asarray(
            np.concatenate([left, i, right]))))

    return permu
