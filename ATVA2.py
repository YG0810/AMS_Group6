"""This Advanced Tactical Voting Analyst considers counter-strategic voting."""

from collections import deque
import numpy as np
from numpy import chararray as npchar
from itertools import permutations
from Types import (
    VoterPreference,
    VotingScheme,
    CandidateResults,
    HappinessMeasure,
    RiskMeasure,
)
from strategy_generators import (
    StrategyGenerator,
    createNDistinctPermutations,
    defaultStrategyGenerator,
)
from voting_schemes import plurality_voting, borda_count_voting
from happiness_measure import NDCG, get_happiness
from risk_measure import FlipRewardRisk, probStrategicVoting

from happiness_measure import *  # NOQA
from risk_measure import *  # NOQA
from voting_schemes import *  # NOQA

ATVA2_Output = tuple[
    CandidateResults, list[float], float, list[set[tuple[list[str], float]]], float
]


class ATVA2:
    def __init__(
        self,
        happiness_measure: HappinessMeasure = lambda _, __, ___, ____: np.nan,
        risk_measure: RiskMeasure = lambda _, __, ___, ____: np.nan,
        strategyGenerator: StrategyGenerator = defaultStrategyGenerator,
    ):
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure
        self.strategyGenerator = strategyGenerator

        self.maxN = 1000

    def analyze(
        self,
        voter_preference: VoterPreference,  # m x n
        voting_scheme: VotingScheme,
    ) -> ATVA2_Output:
        """
        Tree wise approach (for each voter);
        1. Get all strategic voting options, same as the btva
        2. For each strategic voting option, get the counter-strategic voting options.
        3. Calculate what strategic voting options are left for the initial voter that improved the happiness after the counter-strategic voting.
        """
        # Step 1: Get all strategic voting options
        outcome, individual_happiness, overall_happiness, strategic_options, risk = (
            self._single_step_analyse(voter_preference, voting_scheme)
        )

        # Step 2: For each strategic voting option, get the counter-strategic voting options
        counter_voting_risks = []

        for i in range(len(voter_preference[0])):
            for strategic_option, happiness in strategic_options[i]:
                strategic_voter = i
                strategic_voter_original_hapiness = individual_happiness[i]

                if strategic_voter_original_hapiness < happiness:
                    # Adjust the voter preference
                    voter_preference_adj = voter_preference.copy()
                    voter_preference_adj[:, i] = strategic_option

                    # Analyze the counter-strategic voting
                    _, _, _, _, counter_risk = self._single_step_analyse(
                        voter_preference_adj,
                        voting_scheme,
                        excluded_voter=strategic_voter,
                        excluded_voter_preference=voter_preference[:, i],
                        excluded_voter_original_hapiness=strategic_voter_original_hapiness,
                    )

                    # Calculate the risk of the counter-strategic voting
                    counter_voting_risks.append(counter_risk)

        # Adjusted risk
        if counter_voting_risks:
            risk = risk * (1 - np.array(counter_voting_risks).mean())

        return outcome, individual_happiness, overall_happiness, strategic_options, risk

    def _single_step_analyse(
        self,
        voter_preference: VoterPreference,
        voting_scheme: VotingScheme,
        excluded_voter: int | None = None,
        excluded_voter_preference: npchar | None = None,
        excluded_voter_original_hapiness: float | None = None,
    ):
        """
        Analyze the voting preference of a group of voters using a specific voting scheme.

        :param voter_preference: An array of shape (m,n), where m is the number of candidates and n is the number of voters.
        :param voting_scheme: The voting scheme to use.
        :return: a tuple containing the following:
                 - Non-strategic voting outcome (dict)
                 - The happiness level of each voter (list[float])
                 - The overall happiness level (float)
                 - For each voter a (possibly empty) set of strategic-voting options (list[set])
                 - Overall risk of strategic voting for the given input (float)
        """
        m, n = voter_preference.shape

        # Non-strategic voting outcome
        outcome = voting_scheme(voter_preference)  # Sort the outcome by value
        outcome = {
            k: v
            for k, v in sorted(outcome.items(), key=lambda item: item[1], reverse=True)
        }

        # Happiness levels
        individual_happiness = [
            self.happiness_measure(
                voter_preference[:, i],
                list(outcome.keys()),  # type:ignore
            )
            for i in range(n)
        ]
        overall_happiness = sum(individual_happiness)
        # Find all possible permutations of the voter's preference
        all_options = self.strategyGenerator(
            np.char.asarray(voter_preference[:, 0]), 10000
        )

        # Strategic voting options
        strategic_options = []
        for i in range(n):
            if excluded_voter and i == excluded_voter:
                strategic_options.append(set())
                continue

            options = set()
            mod_pref = voter_preference.copy()

            for option in all_options:
                # Check the modified outcome

                mod_pref[:, i] = option
                mod_outcome = voting_scheme(mod_pref)
                mod_outcome = {
                    k: v
                    for k, v in sorted(
                        mod_outcome.items(), key=lambda item: item[1], reverse=True
                    )
                }

                # Check the modified happiness
                mod_happiness = self.happiness_measure(
                    voter_preference[:, i],
                    list(mod_outcome.keys()),  # type:ignore
                )
                """if (
                    mod_happiness > individual_happiness[i]
                ):  # Only consider options that increase happiness
                    # Save (modified preference, modified happiness)"""
                # Bloom change: for probStrategicVoting, I need to know how many samples are considered
                # Every other risk measure have been modified to ignore the bad options

                # Reset the hapiness if the excluded voter is still happier then the original hapiness, since it's not a counter vote
                if excluded_voter is not None:
                    excluded_voter_adj_hapiness = self.happiness_measure(
                        excluded_voter_preference,
                        list(mod_outcome.keys()),  # type:ignore
                    )

                    if excluded_voter_original_hapiness < excluded_voter_adj_hapiness:
                        # Not a counter vote, so reset the happiness, excluding it from the risk measurements
                        mod_happiness = individual_happiness[i]

                options.add((option, mod_happiness))
            strategic_options.append(options)

        risk = self.risk_measure(
            voter_preference,
            voting_scheme,
            individual_happiness,
            strategic_options,
            excluded_voter,
        )

        return outcome, individual_happiness, overall_happiness, strategic_options, risk


def main():
    voter_preference = np.char.array(
        # voters:   1    2    3
        [
            ["B", "A", "C"],  # 1st preference
            ["C", "B", "A"],  # 2nd preference
            ["D", "C", "D"],  # 3rd preference
            ["A", "D", "B"],  # 3rd preference
        ]
    )

    print("Original preferences:")
    print(voter_preference)
    test = ATVA2(
        happiness_measure=NDCG,
        risk_measure=probStrategicVoting,
        strategyGenerator=createNDistinctPermutations,
    )

    outcome, happiness, _, _, risk = test.analyze(
        voter_preference=voter_preference,
        voting_scheme=borda_count_voting,
    )
    print("\n")
    print(happiness)
    print(risk)


if __name__ == "__main__":
    main()
