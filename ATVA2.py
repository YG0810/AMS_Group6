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
    ):
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure

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

        # Strategic voting options
        strategic_options = []
        for i in range(n):
            options = set()
            original_happiness = individual_happiness[i]
            all_options = set(permutations(voter_preference[:, i]))

            # Remove original preference
            all_options.discard(tuple(voter_preference[:, i]))

            for option in all_options:
                # Check the modified outcome
                mod_pref = voter_preference.copy()
                mod_pref[:, i] = option
                mod_outcome = voting_scheme(mod_pref)
                mod_outcome = {
                    k: v
                    for k, v in sorted(
                        mod_outcome.items(), key=lambda item: item[1], reverse=True
                    )
                }

                # Check the modified happiness
                mod_happiness = self.happiness_measure(voter_preference[:, i], list(mod_outcome.keys()))
                if mod_happiness > original_happiness:
                    # Check if another voter can counteract this strategy
                    can_be_countered = False
                    for j in range(n):
                        if j == i:
                            continue  # Skip the same voter

                        all_counter_options = set(permutations(voter_preference[:, j]))
                        all_counter_options.discard(tuple(voter_preference[:, j]))

                        for counter_option in all_counter_options:
                            counter_pref = mod_pref.copy()
                            counter_pref[:, j] = counter_option
                            counter_outcome = voting_scheme(counter_pref)
                            counter_outcome = {
                                k: v
                                for k, v in sorted(
                                    counter_outcome.items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            }

                            counter_happiness = self.happiness_measure(
                                voter_preference[:, i], list(counter_outcome.keys())
                            )
                            if counter_happiness < original_happiness:
                                can_be_countered = True
                                break
                        if can_be_countered:
                            break

                    if not can_be_countered:
                        options.add((option, mod_happiness))

            strategic_options.append(options)

        risk = self.risk_measure(
            voter_preference,
            voting_scheme,
            individual_happiness,
            strategic_options,
        )

        return outcome, individual_happiness, overall_happiness, strategic_options, risk
