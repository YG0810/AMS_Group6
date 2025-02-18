"""Hapiness Measurements"""

import numpy as np


def get_happiness(
    preferences: np.char.chararray,
    outcome: list[str],
    preferenceWeights: list[float] | None = None,
    distanceWeights: list[float] | None = None,
) -> float:
    """
    Calculate the happiness of a voter given their preference and the voting outcome.

    :param voter_preference: The preference of the voter.
    :param voting_outcome: The outcome of the voting scheme.
    :param preference_weights: The weights of the preference.
    :param distance_weights: The weights of the distance between the preference and the outcome.
    :return: The happiness score.
    """
    # Set default weights
    if not preferenceWeights:
        preferenceWeights = [1.0] * len(preferences)
    if not distanceWeights:
        distanceWeights = [1.0] * len(preferences)

    # Measure happiness
    happiness_score = 0
    for preference_idx in range(len(preferences)):
        # Retrieve distance between preference and outcome
        distance = abs(
            preference_idx - outcome.index(preferences[preference_idx])
        )

        # Calculate happiness score
        happiness_score += (
            preferenceWeights[preference_idx] * distanceWeights[distance]
        )

    return happiness_score


""" Example Usage """

# Top-1 Binary happiness measure
preference_weights = [1., 0., 0.]
distance_weights = [1., 0., 0.]

print(
    get_happiness(
        ["A", "B", "C"], ["A", "B", "C"], preference_weights, distance_weights
    )
)

# Bottom-1 Binary happiness measure
preference_weights = [1., 0., 0.]
distance_weights = [0., 0., -1.]

print(
    get_happiness(
        ["A", "B", "C"], ["C", "B", "A"], preference_weights, distance_weights
    )
)
