from math import ceil, floor, log2
from typing import Protocol
from numpy import chararray as npchar
import numpy as np

from Types import (
    VoterPreference,
    VotingScheme,
    CandidateResults,
    HappinessMeasure,
    RiskMeasure,
)  # NOQA

def createRanking(
    preferences: VoterPreference,
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
        result.append(
            distance * preferenceWeights[i] * distanceWeights[distance - 1])

    return result


def DCG(ranking: list[float]):
    result = 0.0

    for i in range(len(ranking)):
        # i starts at one in the equation, but we start at 0, so apply offset
        result += ranking[i] / log2(i + 2)

    return result


def NDCG(
    preferences: VoterPreference,
    outcome: list[str],
    preferenceWeights: list[float] | None = None,
    distanceWeights: list[float] | None = None,
) -> float:

    # No weights specified, let's use weights of 1 for everything
    if preferenceWeights is None:
        preferenceWeights = [1.0] * (ceil(len(preferences)/2)) + [0.0] * (floor(len(preferences)/2))
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
    rankings = createRanking(preferences, outcome,
                             preferenceWeights, distanceWeights)

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
    preferences: VoterPreference,
    outcome: list[str],
    preferenceWeights: list[float] | None = None,
    _: list[float] | None = None,
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
    tau = (concordantPairs - (sum(preferenceWeights) - concordantPairs)) / sum(
        preferenceWeights
    )

    # Normalize to [0, 1]
    return (tau + 1) / 2


def BubbleSortDistance(
    preferences: VoterPreference,
    outcome: list[str],
    preferenceWeights: list[float] | None = None,
    _: list[float] | None = None,
) -> float:
    n = len(preferences)

    # No weights specified, let's use weights of 1 for everything
    if preferenceWeights is None:
        preferenceWeights = [1 for _ in preferences]
    elif len(preferenceWeights) < len(preferences):
        paddingAmount = len(preferences) - len(preferenceWeights)
        preferenceWeights += [0 for _ in range(paddingAmount)]

    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(outcome)
    b = np.argsort(preferences)

    # Calculate weighted disorder
    weighted_disorder = (
        np.logical_or(
            np.logical_and(a[i] < a[j], b[i] > b[j]),
            np.logical_and(a[i] > a[j], b[i] < b[j]),
        )
        * np.array(preferenceWeights)[i]
    ).sum()

    return 1 - (weighted_disorder / (n * (n - 1)))


def get_happiness(
    voter_preference: VoterPreference,
    voting_outcome: list,
    preference_weights: list | None = None,
    distance_weights: list | None = None,
):
    """
    Calculate the happiness of a voter given their preference and the voting outcome.
    (Default) --> Top-1 Binary happiness measurement
    (Alternative) --> Bottom-1 Binary happiness measurement (Set preference_weight = [1, 0, ..., 0] & distance_weights = [0, ..., 0, -1])

    :param voter_preference: The preference of the voter.
    :param voting_outcome: The outcome of the voting scheme.
    :param preference_weights: The weights of the preference.
    :param distance_weights: The weights of the distance between the preference and the outcome.
    :return: The happiness score.
    """
    # Set default weights (Top-1 Binary happiness measurement)
    if not preference_weights:
        preference_weights = [1] + [0] * (len(voter_preference) - 1)
    if not distance_weights:
        distance_weights = [1] + [0] * (len(voter_preference) - 1)

    # Measure happiness
    happiness_score = 0
    for preference_idx in range(len(voter_preference)):
        # Retrieve distance between preference and outcome
        distance = abs(
            preference_idx -
            voting_outcome.index(voter_preference[preference_idx])
        )

        # Calculate happiness score
        happiness_score += (
            preference_weights[preference_idx] * distance_weights[distance]
        )

    return happiness_score


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
