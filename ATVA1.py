import numpy as np
from voting_schemes import plurality_voting
from Types import HappinessMeasure, RiskMeasure, VoterPreference, VotingScheme
from strategy_generators import StrategyGenerator, defaultStrategyGenerator, combinationStrategyGenerator
from happiness_measure import KendallTau
from risk_measure import probStrategicVoting

class ATVA1:
    def __init__(
        self,
        happiness_measure: HappinessMeasure = lambda _, __, ___, ____: np.nan,
        risk_measure: RiskMeasure = lambda _, __, ___, ____: np.nan,
        strategyGenerator: StrategyGenerator = defaultStrategyGenerator,
        maxCollusionSize: int = 2,
        exhaustiveSearch: bool = False, # if False, only search for one viable collusion per voter permutation
        printProgress: bool = False
    ):
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure
        self.strategyGenerator = strategyGenerator
        self.maxCollusionSize = maxCollusionSize
        self.exhaustiveSearch = exhaustiveSearch
        self.printProgress = printProgress

    def analyze(
        self,
        voter_preference: VoterPreference,
        voting_scheme: VotingScheme,
    ):
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
        strategic_options = [self.strategyGenerator(voter_preference[:, i], m) for i in range(n)]
        [strategic_options[i].remove(tuple(voter_preference[:, i])) for i in range(n)] # remove original preference

        # potential collusions
        collusion_permutations = combinationStrategyGenerator(range(n), self.maxCollusionSize)

        collusion_options = []

        counter = 0
        for collusion_candidate in collusion_permutations:
            counter += 1
            if self.printProgress:
                print(f'collusion candidate {counter}/{len(collusion_permutations)}')
            strategic_options_indices = [0 for _ in range(self.maxCollusionSize)]
            modified_preference_matrix = voter_preference.copy()

            # apply first possible strategic option
            for i in collusion_candidate:
                modified_preference_matrix[:, i] = strategic_options[i][0]

            while True:
                # check if current collusion is viable (no voter in collusion is worse off)
                current_outcome = voting_scheme(modified_preference_matrix)
                current_outcome = {
                    k: v
                    for k, v in sorted(current_outcome.items(), key=lambda item: item[1], reverse=True)
                }
                delta_happiness = [ # change in happiness for each voter in collusion
                    self.happiness_measure(
                        voter_preference[:, i],
                        list(current_outcome.keys())
                    ) - individual_happiness[i]
                    for i in collusion_candidate
                ]
                if all([delta > 0 for delta in delta_happiness]): # only consider strictly better
                    collusion_options.append((
                        collusion_candidate,
                        [modified_preference_matrix[:, i].copy() for i in collusion_candidate],
                        delta_happiness
                    ))
                    if not self.exhaustiveSearch:
                        break # stop search

                # apply next possible strategic option
                strategic_options_indices[0] += 1
                for i in range(self.maxCollusionSize):
                    if strategic_options_indices[i] >= len(strategic_options[i]):
                        if i + 1 < self.maxCollusionSize: # carry over
                            strategic_options_indices[i] = 0
                            strategic_options_indices[i+1] += 1
                        else:
                            break # handle this case outside the loop
                    modified_preference_matrix[:, collusion_candidate[i]] = \
                        strategic_options[collusion_candidate[i]][strategic_options_indices[i]]
                if strategic_options_indices[-1] >= len(strategic_options[-1]):
                    break # all possible strategic options have been applied

        # split collusion options into individual strategic options (only for risk calculation)
        strategic_options = []
        for i in range(n):
            options = set()
            for j in collusion_options:
                for index, k in enumerate(j[0]):
                    if k == i:
                        options.add((tuple(j[1][index]), j[2][index] + individual_happiness[i]))
            strategic_options.append(options)
        # calculate risk of strategic voting
        collusion_risk = self.risk_measure(
            voter_preference,
            voting_scheme,
            individual_happiness,
            strategic_options
        )

        return outcome, individual_happiness, overall_happiness, collusion_options, collusion_risk

if __name__ == '__main__':
    from generate_test_cases import generate_test
    import random
    random.seed(0)
    voter_preference = generate_test(num_candidates=3, num_voters=10)

    atva = ATVA1(happiness_measure=KendallTau, risk_measure=probStrategicVoting, exhaustiveSearch=True)
    outcome, individual_happiness, overall_happiness, collusion_options, risk = atva.analyze(voter_preference, plurality_voting)
    # print(collusion_options)
