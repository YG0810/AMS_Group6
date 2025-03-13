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
from strategy_generators import StrategyGenerator, createNDistinctPermutations, defaultStrategyGenerator
from voting_schemes import plurality_voting
from happiness_measure import NDCG, get_happiness
from risk_measure import adjProbStrategicVoting

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
        rationality: bool = True
    ) -> ATVA2_Output:
        """
        Tree wise approach (for each voter);
        1. Get all strategic voting options, same as the btva
        2. For each strategic voting option, get the counter-strategic voting options.
        3. Calculate what strategic voting options are left for the initial voter that improved the happiness after the counter-strategic voting.
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
            np.char.asarray(voter_preference[:, 0]), 10000)

        # Strategic voting options
        strategic_options = []
        for i in range(n):
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

                counter_voters_options = set()

                if (
                    mod_happiness > individual_happiness[i]
                ):  # Only consider options that increase happiness
                    # Find all possible permutations of the voter's preference
                    all__counter_options = self.strategyGenerator(
                        np.char.asarray(mod_pref[:, 0]), 10000)

                    for j in range(n):
                        if j == i:
                            continue  # Skip the same voter
                        
                        counter_options = set()

                        counter_mod_pref = mod_pref.copy()
                        counter_happiness = individual_happiness[j]

                        for counter_option in all__counter_options:
                            # Check the counter modified outcome
                            counter_mod_pref[:, j] = counter_option
                            counter_mod_outcome = voting_scheme(counter_mod_pref)
                            counter_mod_outcome = {
                                k: v
                                for k, v in sorted(
                                    counter_mod_outcome.items(), key=lambda item: item[1], reverse=True,
                                )
                            }

                            # Check the counter modified happiness
                            counter_mod_happiness = self.happiness_measure(
                                mod_pref[:, j],
                                list(counter_mod_outcome.keys()),  # type:ignore
                            )

                            # Check the happiness after the counter voter j options
                            mod_happiness_after_counter = self.happiness_measure(
                                voter_preference[:, i],
                                list(counter_mod_outcome.keys()),  # type:ignore
                            )
                            
                            # Add the counter voter j options
                            counter_options.add((counter_option, mod_happiness_after_counter, counter_mod_happiness, counter_happiness))
                        
                        counter_voters_options.add(frozenset(counter_options))

                options.add((option, mod_happiness, frozenset(counter_voters_options)))
            strategic_options.append(options)
        risk = self.risk_measure(
            voter_preference,
            voting_scheme,
            individual_happiness,
            strategic_options,
        )
        return outcome, individual_happiness, overall_happiness, strategic_options, risk


def main():
    voter_preference = np.char.array(
        # voters:   1    2    3
        [
            ["B", "A", "C", "C", "B"],  # 1st preference
            ["A", "B", "B", "B", "C"],  # 2nd preference
            ["C", "C", "A", "A", "A"],  # 3rd preference
        ]
    )

    print("Original preferences:")
    print(voter_preference)
    test = ATVA2(happiness_measure=NDCG, risk_measure=adjProbStrategicVoting, strategyGenerator=createNDistinctPermutations)

    outcome, happiness, _, _, risk = test.analyze(
        voter_preference=voter_preference, 
        voting_scheme=plurality_voting,
    )
    print("\n")
    print(happiness)
    print(risk)


if __name__ == "__main__":
    main()

