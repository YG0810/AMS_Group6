from dataclasses import dataclass, field
from random import random
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
from risk_measure import NaivePSV
from strategy_generators import StrategyGenerator, createNDistinctPermutations, defaultStrategyGenerator
from voting_schemes import plurality_voting

CoalitionStrategies = list[tuple[tuple[int, ...], list[tuple[str, ...]]]]


@dataclass
class StrategyEntry:
    voting_preference: list[str]
    happiness_score: float

    nonindependent_happiness_scores: list[float] = field(
        default_factory=list[float])

    def avg_nonIndependent_happiness_score(self):
        if (len(self.nonindependent_happiness_scores) == 0):
            return -1

        return sum(self.nonindependent_happiness_scores) / len(self.nonindependent_happiness_scores)


@dataclass
class IndividualStrategies:
    strategies: list[StrategyEntry]
    sumHappinessGained: float

    # Gets a random strategy, with more beneficial strategies being more likely
    def getRandomStrategy(self):
        randomNumber = random() * self.sumHappinessGained

        thres = 0

        for strat in self.strategies:
            thres += strat.happiness_score
            if (randomNumber <= thres):
                return strat

        return self.strategies[0]


StrategyList = list[IndividualStrategies]

ATVA4_Output = tuple[
    CandidateResults, list[float], float, list[set[tuple[list[str], float]]], float
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

        individualStrategies: StrategyList = []

        # For each voter, we determine what strategies are available to them, and their happiness if they are the only
        for i in range(n):
            # Also add the original option in the list, to determine whether they want to stick to their guns
            strategies: list[StrategyEntry] = [
                StrategyEntry(list(voter_preference[:, i]), individual_happiness[i])
            ]
            happinessGainedSum: float = 0
            mod_pref = voter_preference.copy()

            for p in permutations:
                mod_pref[:, i] = p
                mod_outcome = voting_scheme(mod_pref)
                mod_outcome = {
                    k: v
                    for k, v in sorted(
                        mod_outcome.items(), key=lambda item: item[1], reverse=True
                    )
                }

                # Check the modified happiness
                # We are *intentionally* checking the modified outcome against the original preference list
                # This is because ultimately the goal of the voter is to improve the results to better match their original preference
                # If we check against the alternate preference, then it implies that we just want the voter to choose a list that matches others rather than their own
                mod_happiness: float = self.happiness_measure(
                    np.char.asarray(voter_preference[:, i]), list(
                        mod_outcome.keys()), None, None
                )

                happinessGained = max(
                    mod_happiness - individual_happiness[i], 0)
                happinessGainedSum += happinessGained
                strategies.append(StrategyEntry(list(p), happinessGained))

            individualStrategies.append(IndividualStrategies(
                sorted(strategies, key=lambda e: e.happiness_score, reverse=True),
                happinessGainedSum
            ))

        # Now that we have the strategies that are possible, let's make an archaic simulation
        # Each voter have a chance of "attempting" to manipulate the outcome, independently and without consideration to others
        # When they decide to manipulate, they are more likely to select a strategy that benefits them more
        # They may still choose a less-good strategy due to RNG

        # Lets do 10000 simulations
        for i in range(100000):
            cunningVoters: list[tuple[int, StrategyEntry]] = []

            # Determine who is planning to manipulate
            for j in range(n):
                # Voters only have a 30% chance of being a smartass
                if (random() > 0.3):
                    continue

                voterChoice = individualStrategies[j].getRandomStrategy()

                cunningVoters.append((j, voterChoice))

            # Apply the strategies
            mod_pref = voter_preference.copy()
            for v in cunningVoters:
                mod_pref[:, v[0]] = v[1].voting_preference

            mod_outcome = voting_scheme(mod_pref)
            mod_outcome = list({
                k: v
                for k, v in sorted(mod_outcome.items(), key=lambda item: item[1], reverse=True)
            }.keys())

            for v in cunningVoters:
                mod_happiness = self.happiness_measure(np.char.asarray(
                    voter_preference[:, v[0]]), mod_outcome, None, None)

                v[1].nonindependent_happiness_scores.append(mod_happiness)
                pass

        # Re-sort the strategies of each voter based on the average score obtained in a practical setting
        strategiesRes = [
            {(tuple(s.voting_preference), s.avg_nonIndependent_happiness_score())
             for s in IS.strategies if s.avg_nonIndependent_happiness_score() != -1}
            for IS in individualStrategies]

        risk = self.risk_measure(
            voter_preference, voting_scheme, individual_happiness, strategiesRes)

        return outcome, individual_happiness, overall_happiness, strategiesRes, risk  # type:ignore


def main(number: int):
    voter_preference = generate_test(10, 10)

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

    coalitionSize = number

    start_time = time()
    btva = ATVA4(happiness_measure=NDCG, risk_measure=NaivePSV,
                 strategyGenerator=createNDistinctPermutations, maxCoalitionSize=coalitionSize)
    outcome, happiness, overall_happiness, strategies, risk = btva.analyze(
        voter_preference, plurality_voting
    )
    endTimeDefault = time()

    defaultDuration = endTimeDefault - start_time

    print("Risk (1000 distinct permutations)", risk)
    print("Time taken (1000 distinct permutations)", defaultDuration)


if __name__ == "__main__":
    for i in range(1, 6):
        main(i)
