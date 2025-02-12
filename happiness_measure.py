from math import log2
from numpy import chararray as npchar
import numpy as np


def createRanking(
    preferences: npchar,
    outcome: list[str],
    preferenceWeights: list[float],
    distanceWeights: list[float],
) -> list[float]:

    NonZeroWeightsCount = 0
    for w in preferenceWeights:
        if w > 0:
            NonZeroWeightsCount += 1

    result: list[float] = []
    for i, key in enumerate(preferences):
        distance = max(0, NonZeroWeightsCount - outcome.index(key))
        result.append(distance * preferenceWeights[i] * distanceWeights[distance - 1])

    return result


def DCG(ranking: list[float]):
    result = 0.0

    for i in range(len(ranking)):
        # i starts at one in the equation, but we start at 0, so apply offset
        result += ranking[i] / log2(i + 2)

    return result


def NDCG(
    preferences: npchar,
    outcome: list[str],
    preferenceWeights: list[float] | None = None,
    distanceWeights: list[float] | None = None,
) -> float:

    # No weights specified, let's use weights of 1 for everything
    if preferenceWeights is None:
        preferenceWeights = [1.0] * len(preferences)
    elif len(preferenceWeights) < len(preferences):
        paddingAmount = len(preferences) - len(preferenceWeights)
        preferenceWeights += [0] * paddingAmount

    # No weights specified, let's use weights of 1 for everything
    if distanceWeights is None:
        distanceWeights = [1.0] * len(preferences)
    elif len(distanceWeights) < len(preferences):
        paddingAmount = len(preferences) - len(distanceWeights)
        preferenceWeights += [0] * paddingAmount

    # Let's generate the rankings of the preferences
    rankings = createRanking(preferences, outcome, preferenceWeights, distanceWeights)

    # Let's generate the ideal rankings for normalisation, which is possible if pref==outcome
    # Note that I intentionally sacrificed a simple implementation in favour of reusing the same code path for consistency
    idealRankings = createRanking(
        np.char.array(outcome), outcome, preferenceWeights, distanceWeights
    )

    dcg = DCG(rankings)
    idcg = DCG(idealRankings)

    if idcg == 0:
        return 0

    result = dcg / idcg

    return result


def KendallTau(
    preferences: npchar,
    outcome: list[str],
    preferenceWeights: list[float] | None = None,
    _: list[float] | None = None
) -> float:

    # No weights specified, let's use weights of 1 for everything
    if preferenceWeights is None:
        preferenceWeights = [1 for _ in preferences]
    elif len(preferenceWeights) < len(preferences):
        paddingAmount = len(preferences) - len(preferenceWeights)
        preferenceWeights += [0 for _ in range(paddingAmount)]

    # Count the number of concordant pairs
    concordantPairs = 0
    for i in range(len(preferences)):
        if str(preferences[i]) == str(outcome[i]):
            concordantPairs += preferenceWeights[i]

    # Calculate the Kendall Tau measure
    tau = (concordantPairs - (sum(preferenceWeights) - concordantPairs)) / sum(preferenceWeights)

    # Normalize to [0, 1]
    return (tau + 1) / 2


# Bubble Sort Distance function to count swaps
def bubble_sort_distance(arr, candidate):
    swaps = 0
    n = len(arr)
    arr = list(arr)  # Convert to list to perform bubble sort
    
    # first locations
    index = arr.index(candidate)

    #  Move the candidate to the first position using Bubble Sort swaps
    while index > 0:
        #Swap
        arr[index], arr[index - 1] = arr[index - 1], arr[index]  
        swaps += 1
        index -= 1

    return swaps

#  Extract unique candidates (A, B, C)
candidates = np.unique(voter_preference)  
swap_counts = {str(candidate): 0 for candidate in candidates}  

# Process each voter (each column in the matrix)
for voter in range(voter_preference.shape[1]):
    preferences = voter_preference[:, voter]  # preferences of a voter

    # calculate swap
    for candidate in candidates:
        swap_counts[str(candidate)] += bubble_sort_distance(preferences, candidate)

# print result
print("{", ", ".join(f"'{key}': {value}" for key, value in swap_counts.items()), "}")



# Test code
def testPerfectChoices(n: int, k: int):
    pref = np.char.array([str(i) for i in range(n)])
    outcome = list(pref)

    # We only consider top k choices
    preferenceWeights = [1.0] * k

    result = NDCG(pref, outcome, preferenceWeights)

    assert result == 1

    print(result)


def testCompletelyFucked(n: int):
    pref = np.char.array([str(i) for i in range(n)])
    outcome = list(reversed(pref))

    # We only consider top k choices
    preferenceWeights = [1.0] * int(n / 2)

    result = NDCG(pref, outcome, preferenceWeights)

    assert result == 0

    print(result)


def testCompletelyFuckedAllChoicesConsidered(n: int):
    pref = np.char.array([str(i) for i in range(n)])
    outcome = list(reversed(pref))

    result = NDCG(pref, outcome)

    assert result > 0 and result < 1

    print(result)


def main():
    testPerfectChoices(100, 100)
    testCompletelyFucked(100)
    testCompletelyFuckedAllChoicesConsidered(100)


if __name__ == "__main__":
    main()
