from math import log2
from numpy import chararray as npchar
import numpy as np


def createRanking(
    preferences: npchar,
    outcome: list[str],
    preferenceWeights: list[float],
    distanceWeights: list[float],
) -> list[float]:

    result: list[float] = []
    for i, key in enumerate(preferences):
        distance = len(outcome) - outcome.index(key)
        result.append(distance * preferenceWeights[i] * distanceWeights[distance - 1])

    return result


def DCG(ranking: list[float]):
    result = 0.0

    for i in range(len(ranking)):
        # i starts at one in the equation, but we start at 0, so apply offset
        result += ranking[i] / log2(i + 2)

    return result


def iDCG(ranking: list[float]):
    ranking = sorted(ranking, reverse=True)

    return DCG(ranking)


def NDCG(
    preferences: npchar,
    outcome: list[str],
    preferenceWeights: list[float] | None = None,
    distanceWeights: list[float] | None = None,
) -> float:

    # No weights specified, let's use weights of 1 for everything
    if preferenceWeights is None:
        preferenceWeights = [1 for _ in preferences]

    # No weights specified, let's use weights of 1 for everything
    if distanceWeights is None:
        distanceWeights = [1 for _ in preferences]

    rankings = createRanking(preferences, outcome, preferenceWeights, distanceWeights)

    result = DCG(rankings) / iDCG(rankings)

    return result


# Test code
def main():

    pref = np.char.array(["A", "B", "D", "E", "F"])
    outcome = ["A", "B", "F", "D", "E"]

    print(NDCG(pref, outcome))
    pass


if __name__ == "__main__":
    main()
