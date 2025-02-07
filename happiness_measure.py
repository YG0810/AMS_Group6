from math import log2
from numpy import chararray as npchar
import numpy as np


def createRanking(preferences: npchar, outcome: list[str]) -> list[int]:
    return [
        (len(outcome) - outcome.index(key)) if key in outcome else 0
        for key in preferences
    ]


def DCG(ranking: list[int]):
    result = 0.0

    for i in range(len(ranking)):
        # i starts at one in the equation, but we start at 0, so apply offset
        result += ranking[i] / log2(i + 2)

    return result


def iDCG(ranking: list[int]):
    ranking = sorted(ranking, reverse=True)

    return DCG(ranking)


def NDCG(
    preferences: npchar, outcome: list[str], weights: list[float] | None = None
) -> float:
    # No weights specified, let's use weights of 1 for everything
    if weights is None:
        weights = [1 for _ in preferences]

    rankings = createRanking(preferences, outcome)

    result = DCG(rankings) / iDCG(rankings)

    return result


def main():

    pref = np.char.array(["A", "B", "D", "E", "F"])
    outcome = ["A", "B", "F", "D", "E"]
    rankings = createRanking(pref, outcome)

    print(DCG(rankings))
    pass


if __name__ == "__main__":
    main()
