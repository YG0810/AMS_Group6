import itertools
from time import time
import numpy as np
from Types import (
    CandidateResults,
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

CoalitionStrategies = list[tuple[tuple[int, ...], list[tuple[str, ...]]]]

ATVA4_Output = tuple[
    CandidateResults, list[float], float, CoalitionStrategies, float
]


class ATVA4:
    def __init__(
        self,
        happiness_measure: HappinessMeasure = lambda _, __, ___, ____: np.nan,
        risk_measure: RiskMeasure = lambda _, __, ___, ____: np.nan,
        strategyGenerator: StrategyGenerator = defaultStrategyGenerator,
        maxCoalitionSize: int = 4
    ):
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure
        self.strategyGenerator = strategyGenerator
        self.maxCoalitionSize = maxCoalitionSize

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
        permutations = self.strategyGenerator(
            np.char.asarray(voter_preference[:, 0]), 10000)

        potential_coalitions: list[tuple[int, ...]] = []
        for i in range(1, self.maxCoalitionSize+1):
            potential_coalitions += itertools.combinations(range(0, n), i)

        coalitionStrategies: CoalitionStrategies = []

        for cI in range(len(potential_coalitions)):
            coalition = potential_coalitions[cI]
            coalitionChoices = [0] * len(coalition)

            mod_pref = voter_preference.copy()

            while (True):
                # Apply modified preferences
                for i in range(len(coalition)):
                    mod_pref[:, coalition[i]
                             ] = permutations[coalitionChoices[i]]

                mod_outcome = voting_scheme(mod_pref)
                mod_outcome = {
                    k: v
                    for k, v in sorted(
                        mod_outcome.items(), key=lambda item: item[1], reverse=True
                    )
                }

                # Check the modified happiness
                groupHappinessGained = 0.0

                for i in range(len(coalition)):
                    mod_happiness = self.happiness_measure(
                        permutations[coalitionChoices[i]], list(
                            mod_outcome.keys())  # type:ignore
                    )

                    originalhappiness = individual_happiness[coalition[i]]
                    groupHappinessGained += mod_happiness - originalhappiness\


                if (groupHappinessGained > 0):
                    coalitionStrategies.append(
                        (tuple(coalition), [permutations[i] for i in coalitionChoices]))

                # Increment and carry
                carry = True
                for i in range(len(coalition)):
                    if not carry:
                        break

                    carry = False
                    coalitionChoices[i] += 1
                    if (coalitionChoices[i] >= len(permutations)):
                        coalitionChoices[i] = 0
                        carry = True

                if (carry):
                    break

        """ risk = self.risk_measure(
            voter_preference,
            voting_scheme,
            individual_happiness,
            strategic_options,
        ) """

        return outcome, individual_happiness, overall_happiness, coalitionStrategies, 0


def main(number:int):
    voter_preference = generate_test(5, 5)

    """ start_time = time()
    btva = ATVA4(happiness_measure=NDCG, risk_measure=probStrategicVoting)
    outcome, happiness, overall_happiness, _, risk = btva.analyze(
        voter_preference, plurality_voting
    )
    endTimeDefault = time()

    defaultDuration = endTimeDefault - start_time

    print("Outcome ", outcome)
    print("Happiness", happiness)
    print("Risk (default strategy generation)", risk)
    print("Time taken (default strategy generation)", defaultDuration) """

    coalitionSize= number

    start_time = time()
    btva = ATVA4(happiness_measure=NDCG, risk_measure=probStrategicVoting,
                 strategyGenerator=createNDistinctPermutations, maxCoalitionSize=coalitionSize)
    outcome, happiness, overall_happiness, coalitionStrategies, risk = btva.analyze(
        voter_preference, plurality_voting
    )
    endTimeDefault = time()

    defaultDuration = endTimeDefault - start_time

    print("Risk (1000 distinct permutations)", risk)
    print("Time taken (1000 distinct permutations)", defaultDuration)
    print(f"Coalition strategies count (coalition size {coalitionSize}): ",
          len(coalitionStrategies))


if __name__ == "__main__":
    for i in range(1, 6):
        main(i)
