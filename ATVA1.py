import numpy as np
from voting_schemes import plurality_voting
from Types import HappinessMeasure, RiskMeasure, VoterPreference, VotingScheme
from strategy_generators import StrategyGenerator, defaultStrategyGenerator
from happiness_measure import KendallTau

class ATVA1:
    def __init__(
        self,
        happiness_measure: HappinessMeasure = lambda _, __, ___, ____: np.nan,
        risk_measure: RiskMeasure = lambda _, __, ___, ____: np.nan,
        strategyGenerator: StrategyGenerator = defaultStrategyGenerator,
        maxCollusionSize: int = 2,
        exhaustiveSearch: bool = False # if False, only search for one viable collusion per voter permutation
    ):
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure
        self.strategyGenerator = strategyGenerator
        self.maxCollusionSize = maxCollusionSize
        self.exhaustiveSearch = exhaustiveSearch

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
        strategic_options = [self.strategyGenerator(voter_preference[:, i], m)[1:] for i in range(n)]

        # potential collusions
        collusion_permutations_raw = self.strategyGenerator(range(n), self.maxCollusionSize)
        # remove mirrored permutations
        collusion_permutations = []
        for perm in collusion_permutations_raw:
            if perm[::-1] not in collusion_permutations:
                collusion_permutations.append(perm)

        collusion_options = []

        counter = 0
        for collusion_candidate in collusion_permutations:
            counter += 1
            print(f'collusion candidate {counter}/{len(collusion_permutations)}')
            strategic_options_indices = [0 for _ in range(self.maxCollusionSize)]
            modified_preference_matrix = voter_preference.copy()

            # apply first possible strategic option, original not allowed since it wouldn't be collusion
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
                        modified_preference_matrix[:, i],
                        list(current_outcome.keys())
                    ) - individual_happiness[i]
                    for i in collusion_candidate
                ]
                if all([delta > 0 for delta in delta_happiness]): # only consider strictly better
                    collusion_options.append((
                        collusion_candidate,
                        [modified_preference_matrix[:, i] for i in collusion_candidate],
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

        # collusion risk calculated elsewhere
        collusion_risk = np.nan

        return outcome, individual_happiness, overall_happiness, collusion_options, collusion_risk

if __name__ == '__main__':
    voter_preference = np.char.array(
    # voters: 1    2    3    4
        [
            ["B", "A", "C", "C"],  # 1st preference
            ["C", "C", "B", "B"],  # 2nd preference
            ["A", "B", "A", "A"]   # 3rd preference
        ]
    )

    atva = ATVA1(happiness_measure=KendallTau, exhaustiveSearch=True)
    outcome, individual_happiness, overall_happiness, collusion_options, __ = atva.analyze(voter_preference, plurality_voting)
    # print collusion options for (1, 3)
    print(len(collusion_options))
