import numpy as np
from Types import (
    VoterPreference,
    VotingScheme,
    CandidateResults,
    HappinessMeasure,
    RiskMeasure,
)
from strategy_generators import StrategyGenerator, defaultStrategyGenerator

# (non-strategic voting outcome, voter happiness, overall happiness, voting options per voter, overall risk)
BTVA_Output = tuple[
    CandidateResults, list[float], float, list[set[tuple[list[str], float]]], float
]


class BTVA:
    def __init__(
        self,
        happiness_measure: HappinessMeasure = lambda _, __, ___, ____: np.nan,
        risk_measure: RiskMeasure = lambda _, __, ___, ____: np.nan,
        strategyGenerator: StrategyGenerator = defaultStrategyGenerator
    ):
        """
        Create a Basic Tactical Voting Analyst (BTVA) object.

        :param happiness_measure: A function that measures the happiness of a voter given their preference and the outcome.
        :param risk_measure: A function that measures the risk of strategic voting for a given input.
        """
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure
        self.strategy_generator = strategyGenerator

    def analyze(
        self,
        voter_preference: VoterPreference,
        voting_scheme: VotingScheme,
    ) -> BTVA_Output:
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
        all_options = self.strategy_generator(
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
                """if (
                    mod_happiness > individual_happiness[i]
                ):  # Only consider options that increase happiness
                    # Save (modified preference, modified happiness)"""
                # Bloom change: for probStrategicVoting, I need to know how many samples are considered
                # Every other risk measure have been modified to ignore the bad options
                options.add((option, mod_happiness))
            strategic_options.append(options)
        risk = self.risk_measure(
            voter_preference,
            voting_scheme,
            individual_happiness,
            strategic_options,
        )

        return outcome, individual_happiness, overall_happiness, strategic_options, risk
