import numpy as np
from itertools import product, combinations
from Types import (
    VoterPreference,
    VotingScheme,
)


def FlipRewardRisk(
    voter_preference: VoterPreference,
    _: VotingScheme | None,
    individual_happiness: list[float],
    strategic_options: list,
) -> float:
    """
    Computes the likelihood of strategic voting by evaluating the trade-off between preference changes
    and potential happiness gains. For each voter, analyzes how likely they are to deviate from their
    true preferences when presented with strategic options that could increase their happiness.

    The risk score (between 0 and 1) reflects how tempting a strategic option is, based on:
    1. How much happiness could be gained (delta happiness)
    2. How many preference switches are needed (inversion distance)
    Higher scores suggest higher likelihood of strategic voting.

    Args:
        voter_preference (np.ndarray): Original preference rankings for each voter
        individual_happiness (List[float]): Current happiness values for each voter (must be floats between 0 and 1, meaning normalized)
        strategic_options (list): List of possible strategic options for each voter,
                                where each option is a tuple (preference_ranking, expected_happiness)
        p (float): Sensitivity parameter controlling risk assessment, must be in range [1.3, 1.7]
        - p=1.3 : Conservative assessment (preferably)
            Assigns high risk mainly when large happiness gains require few preference changes
        - p=1.7 : Stringent assessment 
            More likely to flag subtle strategic opportunities
            Assumes voters are tempted even by small happiness gains if few changes needed

    Returns:
        tuple: (risks, overall_max_risk) where:
            - risks: List of (preference, risk_score) tuples for each voter's highest risk option
            - overall_max_risk: Maximum risk score across all voters (float between 0 and 1)

    Raises:
        ValueError: If p is not in the range [1.3, 1.7]
    """

    p = 1.3  # define manually
    if not 1.3 <= p <= 1.7:
        raise ValueError(f"Parameter p must be in range [1.3, 1.7], got {p}")

    risks = [(None, 0.0)]
    for i, options in enumerate(strategic_options):
        if not options:
            continue

        risks4i = [(None, 0.0)]
        for pref in options:
            # Ignore options that are not good
            if (pref[1] <= individual_happiness[i]):
                continue

            preference, happiness = pref
            norm_dist = inversion_ranking_distance(
                np.char.array(voter_preference[:, i]), list(preference)
            )
            delta_happ = abs(happiness - individual_happiness[i])
            score = np.tanh(delta_happ / np.log(norm_dist ** (p - 1) + 1))
            risks4i.append((preference, score))

        max_risk_option = max(risks4i, key=lambda x: x[1])
        risks.append(max_risk_option)
    overall_max_risk = max(risks, key=lambda x: x[1])[1]

    # return risks, overall_max_risk
    return overall_max_risk


def JointFlipRewardRisk(
    voter_preference: VoterPreference,
    _: VotingScheme | None,
    individual_happiness: list[float],
    strategic_options: list,
) -> float:
    """
    Computes the joint likelihood of strategic voting by evaluating all possible
    combinations of honest/strategic voting among voters.

    Args:
        voter_preference (np.ndarray): Original preference rankings for each voter
        individual_happiness (List[float]): Current happiness values for each voter
        strategic_options (list): List of possible strategic options for each voter
        p (float): Sensitivity parameter controlling risk assessment, must be in range [1.3, 1.7]

    Returns:
        - overall_max_risk: Maximum risk score (float between 0 and 1)

    """

    p = 1.3  # define manually
    if not 1.3 <= p <= 1.7:
        raise ValueError(f"Parameter p must be in range [1.3, 1.7], got {p}")

    individual_risks = []
    for i, options in enumerate(strategic_options):
        if not options:
            individual_risks.append(0.0)
            continue

        risks4i = []
        for pref in options:
            # Ignore options that are not good
            if (pref[1] <= individual_happiness[i]):
                continue

            preference, happiness = pref
            norm_dist = inversion_ranking_distance(
                np.char.array(voter_preference[:, i]), list(preference)
            )
            delta_happ = abs(happiness - individual_happiness[i])
            score = np.tanh(delta_happ / np.log(norm_dist ** (p - 1) + 1))
            risks4i.append(score)

        individual_risks.append(max(risks4i))

    n_voters = len(individual_risks)
    scenarios = list(product([0, 1], repeat=n_voters))

    overall_max_risk = 0.0
    for scenario in scenarios:
        if all(x == 0 for x in scenario):  # all voters vote honestly
            continue

        prob = 1.0
        for voter_idx, is_strategic in enumerate(scenario):
            if (
                individual_risks[voter_idx] == 0.0
            ):  # a voter has individual risk 0.0, hence votes honestly guaranteed
                continue
            if is_strategic:
                prob *= individual_risks[voter_idx]
            else:
                prob *= 1 - individual_risks[voter_idx]
        overall_max_risk = max(overall_max_risk, prob)

    return overall_max_risk


def inversion_ranking_distance(base_pref: VoterPreference, option_pref: list) -> float:
    """
    Computes the normalized minimal rearrangement distance between two rankings based on the number of
    displacements required to transform one ranking into another. The result is normalized between 0 and 1,
    where 0 means the rankings are identical and 1 means they are completely reversed.

    Args:
        base_pref (list): The reference ranking (initial preference of a candidate)
        option_pref (list): The ranking to be rearranged to match the base ranking

    Returns:
        float: Normalized minimal rearrangement distance between 0 and 1

    Raises:
        ValueError: If rankings have different lengths
    """
    if len(base_pref) != len(option_pref):
        raise ValueError("Rankings must have equal length")

    target_positions = {value: i for i, value in enumerate(base_pref)}
    current_positions = [target_positions[value] for value in option_pref]

    inversions = 0
    n = len(current_positions)
    for i in range(n):
        for j in range(i + 1, n):
            if current_positions[i] > current_positions[j]:
                inversions += 1

    max_inversions = (n * (n - 1)) // 2
    return inversions / max_inversions if max_inversions > 0 else 0.0


def probStrategicVoting(
    _: np.ndarray,
    __: VotingScheme | None,
    voterHappiness: list[float],
    strategicOptions: list[set[tuple[np.chararray, float]]],
):
    n = len(voterHappiness)
    count = 0.0

    for i, o in enumerate(strategicOptions):
        for o2 in o:
            if o2[1] > voterHappiness[i]:
                count += 1
                break

    return count / n


def WinnerChangeRisk(
    voter_preference: VoterPreference,
    voting_scheme: VotingScheme,
    individual_happiness: list[float],
    strategic_options: list,
) -> float:
    """
    Calculate the proportion of strategic votes that successfully change the winner.

    Args:
        voter_preference (np.ndarray): Original preference rankings for each voter
        voting_scheme: Function that returns a dict mapping candidates to scores.
        strategic_options (list): List of possible strategic options for each voter,
                                where each option is a tuple (preference_ranking, expected_happiness)

    Returns:
        - overall_max_risk: Maximum risk score (float between 0 and 1)

    """
    if not strategic_options:
        return 0.0

    true_results = voting_scheme(voter_preference)
    true_winner = min(true_results.items(), key=lambda x: (-x[1], x[0]))[0]
    # true_leaderboard = [name for name, _ in sorted(true_results.items(), key=lambda item: (-item[1], item[0]))]

    risks = []
    for voter_idx, voter_opts in enumerate(strategic_options):

        successful = 0
        total = 0
        modified_prefs = voter_preference.copy()

        for opt in voter_opts:
            # Ignore options that are not good
            if (opt[1] <= individual_happiness[voter_idx]):
                continue

            total += 1
            modified_prefs[:, voter_idx] = opt[0]
            new_results = voting_scheme(modified_prefs)
            new_winner = min(new_results.items(),
                             key=lambda x: (-x[1], x[0]))[0]
            # new_leaderboard = [name for name, _ in sorted(new_results.items(), key=lambda item: (-item[1], item[0]))]

            if new_winner != true_winner:
                # if new_leaderboard != true_leaderboard:
                successful += 1
                continue
        risks.append(successful / total if total > 0 else 0.0)

    return max(risks)


def CollusionChangeRisk(
    voter_preference: VoterPreference,
    voting_scheme: VotingScheme,
    invididual_happiness: list[float],
    strategic_options: list,
) -> float:
    """
    Calculate the proportion of strategic manipulations (single-voter or two-voter collusions) that successfully change the winner.

    Args:
    voter_preference (np.ndarray): Original preference rankings for each voter.
    voting_scheme (callable): Function that returns a dict mapping candidates to scores.
    strategic_options (list): List of possible strategic options for each voter,
                            where each option is a tuple (preference_ranking, expected_happiness).

    Returns:
    collusion_risk (float): The proportion of successful manipulations (0 to 1) that change the winner.
    """
    if not strategic_options:
        return 0.0

    true_results = voting_scheme(voter_preference)
    true_winner = min(true_results.items(), key=lambda x: (-x[1], x[0]))[0]

    def test_manipulation(voter_indices, strategic_choices):
        modified_prefs = voter_preference.copy()
        for voter_idx, choice in zip(voter_indices, strategic_choices):
            modified_prefs[:, voter_idx] = choice[0]

        new_results = voting_scheme(modified_prefs)
        new_winner = min(new_results.items(), key=lambda x: (-x[1], x[0]))[0]
        return new_winner != true_winner

    successful_attempts = 0
    total_attempts = 0

    # 1. Single Voter Manipulation
    for voter_idx, options in enumerate(strategic_options):

        for choice in options:
            # Ignore options that are not good
            if (choice[1] <= invididual_happiness[voter_idx]):
                continue

            total_attempts += 1
            if test_manipulation([voter_idx], [choice]):
                successful_attempts += 1

    # 2. Two-Voter Collusion
    for voter_indices in combinations(range(len(strategic_options)), 2):

        voter_options = [strategic_options[i] for i in voter_indices]
        choices_for_voter_1 = [choice for choice, eh in voter_options[0]
                               if eh > invididual_happiness[voter_indices[0]]]
        choices_for_voter_2 = [choice for choice, eh in voter_options[1]
                               if eh > invididual_happiness[voter_indices[1]]]

        for coalition_choices in combinations(
            choices_for_voter_1 + choices_for_voter_2, 2
        ):
            total_attempts += 1
            if test_manipulation(voter_indices, coalition_choices):
                successful_attempts += 1

    collusion_risk = successful_attempts / \
        total_attempts if total_attempts > 0 else 0.0
    return collusion_risk
