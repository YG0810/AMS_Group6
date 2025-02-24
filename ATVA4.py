from random import shuffle
from time import time
from timeit import Timer
from typing import Any
import numpy as np
from BTVA import BTVA_Output
from Types import (
    VoterPreference,
    VotingScheme,
    HappinessMeasure,
    RiskMeasure,
)
from generate_test_cases import generate_test
from happiness_measure import NDCG
from risk_measure import probStrategicVoting
from strategy_generators import StrategyGenerator, createNDistinctPermutations, defaultStrategyGenerator
from voting_schemes import plurality_voting

ATVA4_Output = BTVA_Output


class ATVA4:
    def __init__(
        self,
        happiness_measure: HappinessMeasure = lambda _, __, ___, ____: np.nan,
        risk_measure: RiskMeasure = lambda _, __, ___, ____: np.nan,
        strategyGenerator: StrategyGenerator = defaultStrategyGenerator
    ):
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure
        self.strategyGenerator = strategyGenerator

    def analyze(
        self,
        voter_preference: VoterPreference,
        voting_scheme: VotingScheme,
    ) -> ATVA4_Output:

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
                voter_preference[:, i], list(outcome.keys())  # type:ignore
            )
            for i in range(n)
        ]

        overall_happiness = sum(individual_happiness)

        # Strategic voting options
        strategic_options = []
        for i in range(n):
            options = set()
            mod_pref = voter_preference.copy()

            # Find all possible permutations of the voter's preference
            permutations = self.strategyGenerator(
                np.char.asarray(voter_preference[:, i].flatten()), 10000)

            for option in permutations:
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
                    option, list(mod_outcome.keys())  # type:ignore
                )
                if (
                    mod_happiness > individual_happiness[i]
                ):  # Only consider options that increase happiness
                    # Save (modified preference, modified happiness)
                    options.add((option, mod_happiness))
            strategic_options.append(options)
        risk = self.risk_measure(
            voter_preference,
            voting_scheme,
            individual_happiness,
            strategic_options,
        )

        return outcome, individual_happiness, overall_happiness, strategic_options, risk


def main():
    voter_preference = generate_test(10, 10)

    start_time = time()
    btva = ATVA4(happiness_measure=NDCG, risk_measure=probStrategicVoting)
    outcome, happiness, overall_happiness, _, risk = btva.analyze(
        voter_preference, plurality_voting
    )
    endTimeDefault = time()

    defaultDuration = endTimeDefault - start_time

    print("Outcome ", outcome)
    print("Happiness", happiness)
    print("Risk (default strategy generation)", risk)
    print("Time taken (default strategy generation)", defaultDuration)

    start_time = time()
    btva = ATVA4(happiness_measure=NDCG, risk_measure=probStrategicVoting, strategyGenerator=createNDistinctPermutations)
    outcome, happiness, overall_happiness, _, risk = btva.analyze(
        voter_preference, plurality_voting
    )
    endTimeDefault = time()

    defaultDuration = endTimeDefault - start_time

    print("Risk (1000 distinct permutations)", risk)
    print("Time taken (1000 distinct permutations)", defaultDuration)


if __name__ == "__main__":
    main()
