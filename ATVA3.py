from collections import deque
import numpy as np
from numpy import chararray as npchar
from itertools import permutations
from Types import (
    VotingScheme,
    HappinessMeasure,
    RiskMeasure,
    VoterPreference
)

from happiness_measure import *  # NOQA
from risk_measure import *  # NOQA
from voting_schemes import *  # NOQA

# (reconstructed voter preference, non-strategic voting outcome, voter happiness, overall happiness, voting options per voter, overall risk)
ATVA3_Output = tuple[
    VoterPreference,
    dict[str, int],
    list[float],
    float,
    list[set[tuple[list[str], float]]],
    float,
]


class ATVA3:
    def __init__(
        self,
        happiness_measure: HappinessMeasure = lambda _, __, ___, ____: np.nan,
        risk_measure: RiskMeasure = lambda _, __, ___, ____: np.nan,
    ):
        """
        Create a Basic Tactical Voting Analyst (BTVA) object.

        :param happiness_measure: A function that measures the happiness of a voter given their preference and the outcome.
        :param risk_measure: A function that measures the risk of strategic voting for a given input.
        """
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure

    def analyze(
        self, voter_preference: npchar, voting_scheme: VotingScheme, sim_params: dict
    ) -> ATVA3_Output:
        """
        Analyze the voting preference of a group of voters using a specific voting scheme.

        :param voter_preference: An array of shape (m,n), where m is the number of candidates and n is the number of voters.
        :param voting_scheme: The voting scheme to use.
        :param sim_params: simulation parameters.
        :return: a tuple containing the following:
                 - Reconstructed voter preferences from election leaderbord (npchar)
                 - Non-strategic voting outcome (dict)
                 - The happiness level of each voter (list[float])
                 - The overall happiness level (float)
                 - For each voter a (possibly empty) set of strategic-voting options (list[set])
                 - Overall risk of strategic voting for the given input (float)
        """
        m, n = voter_preference.shape

        # Reconstruct voter preference based on leaderboard of true outcome
        attempt = 0
        reconstruct_preference = np.empty((m, n), dtype="U1")
        while not self._is_valid_preference_matrix(reconstruct_preference):
            attempt *= 1
            reconstruct_preference = self.preference_reconstruct(
                voter_preference=voter_preference,
                voting_scheme=voting_scheme,
                num_simulations=sim_params["num_simulations"],
                window_size=sim_params["window_size"],
                stable_window=sim_params["stable_window"],
                target_acceptance=sim_params["target_acceptance"],
                improvement_threshold=sim_params["improvement_threshold"],
            )

            if attempt > 3:
                print(
                    f"Failed to reconstruct polling for {voting_scheme.__name__} after 3 attempts."
                )
                empty_preference = np.char.array(np.empty((m, n), dtype="U1"))
                return (
                    empty_preference,  # empty preference matrix
                    {},  # empty outcome dictionary
                    [0.0] * n,  # zero happiness for each voter
                    0.0,  # zero overall happiness
                    # empty strategic options for each voter
                    [set() for _ in range(n)],
                    0.0,  # zero risk
                )

        # format reconstruct outcome
        temp = np.empty((m, n), dtype="U1")
        remaining_columns = [col for col in reconstruct_preference.T]
        for i, col in enumerate(voter_preference.T):
            for j, r_col in enumerate(remaining_columns):
                if np.array_equal(col, r_col):
                    temp[:, i] = r_col
                    remaining_columns.pop(j)
                    break
        for i in range(n):
            if np.all(temp[:, i] == ""):
                temp[:, i] = remaining_columns.pop(0)
        reconstruct_preference = np.char.array(temp)

        reconstruct_outcome = voting_scheme(reconstruct_preference)
        reconstruct_outcome = {
            k: v
            for k, v in sorted(
                reconstruct_outcome.items(), key=lambda item: item[1], reverse=True
            )
        }

        # Happiness levels
        individual_happiness = [
            self.happiness_measure(
                reconstruct_preference[:, i],
                list(reconstruct_outcome.keys()),  # type:ignore
            )
            for i in range(n)
        ]
        overall_happiness = sum(individual_happiness)

        # Strategic voting options
        strategic_options = []
        for i in range(n):
            options = set()

            # Find all possible permutations of the voter's preference
            all_options = set(permutations(reconstruct_preference[:, i]))
            all_options.discard(
                tuple(reconstruct_preference[:, i])
            )

            for option in all_options:
                # Check the modified outcome
                mod_pref = reconstruct_preference.copy()
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
                    reconstruct_preference[:, i], list(mod_outcome.keys())  # type:ignore
                )
                if (
                    mod_happiness > individual_happiness[i]
                ):  # Only consider options that increase happiness
                    # Save (modified preference, modified happiness)
                    options.add((option, mod_happiness))
            strategic_options.append(options)
        risk = self.risk_measure(
            reconstruct_preference,
            voting_scheme,
            individual_happiness,
            strategic_options,
        )

        return (
            reconstruct_preference,
            reconstruct_outcome,
            individual_happiness,
            overall_happiness,
            strategic_options,
            risk,
        )

    def preference_reconstruct(
        self,
        voter_preference: npchar,
        voting_scheme: VotingScheme,
        num_simulations: int = 1000,
        window_size: int = 50,
        stable_window: int = 100,
        target_acceptance: float = 0.3,
        improvement_threshold: float = 0.01,
    ) -> npchar:
        """
        Perform Markov Chain Monte Carlo (MCMC) reconstruction of voter preferences. The voter preferences are supposed to be
        reconstructed based on the leaderboard (as typically available as polling data prior to elections) of the candidates.

        :param voter_preference: An array of shape (m, n), where m is the number of candidates and n is the number of voters.
        :param voting_scheme: A callable function that computes the voting outcome.
        :param num_simulations: The number of MCMC simulations to run (default: 1000).
        :param window_size: The size of the acceptance tracking window (default: 50).
        :param stable_window: The number of iterations to consider stability (default: 100).
        :param target_max_acceptance: The target maximum acceptance rate (default: 0.3).
        :param improvement_threshold: The threshold to determine convergence (default: 0.01).
        :return: An array representing the reconstructed voter preference matrix (npchar).
        """

        m, n = voter_preference.shape

        # leaderbord from deterministic voting outcome, not "visible" to ATVA-3
        true_outcome = voting_scheme(voter_preference)
        true_outcome = {
            k: v
            for k, v in sorted(
                true_outcome.items(), key=lambda item: item[1], reverse=True
            )
        }
        leaderboard = [
            name
            for name, _ in sorted(
                true_outcome.items(), key=lambda item: (-item[1], item[0])
            )
        ]

        def initialize_preference():
            pref = np.empty((m, n), dtype="U1")
            candidates = np.array(leaderboard)
            for v in range(n):
                np.random.shuffle(candidates)
                pref[:, v] = candidates
            return pref

        def reconstruct_match(voter_preference):
            votes_distribution = voting_scheme(voter_preference)
            simulated_leaderboard = sorted(
                votes_distribution, key=votes_distribution.get, reverse=True
            )
            return sum(1 for a, b in zip(leaderboard, simulated_leaderboard) if a == b)

        def preference_stability(current_pref, history):
            if len(history) < 2:
                return 0.0
            return np.mean(
                [
                    np.mean(
                        [
                            KendallTau(
                                preferences=current_pref[:,
                                                         v], outcome=hist_pref[:, v]
                            )
                            for v in range(n)
                        ]
                    )
                    for hist_pref in history
                ]
            )

        def acceptance_rate(window):
            return sum(window) / len(window) if window else 1.0

        best_preference = None
        best_score = -1
        current_preference = initialize_preference()
        current_score = reconstruct_match(current_preference)

        acceptance_window = deque(maxlen=window_size)
        preference_history = deque(maxlen=stable_window)
        stability_counter = 0
        phase = 1

        for _ in range(window_size):
            new_preference = current_preference.copy()
            v = np.random.randint(n)
            i, j = np.random.choice(m, 2, replace=False)
            new_preference[i, v], new_preference[j, v] = (
                new_preference[j, v],
                new_preference[i, v],
            )

            new_score = reconstruct_match(new_preference)
            accepted = new_score > current_score or np.random.rand() < np.exp(
                (new_score - current_score)
            )

            if accepted:
                current_preference = new_preference
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_preference = current_preference.copy()

            acceptance_window.append(accepted)

        initial_acceptance_rate = acceptance_rate(acceptance_window)
        if initial_acceptance_rate <= target_acceptance:
            stability = preference_stability(
                current_preference, preference_history)
            return current_preference, stability

        while phase <= 3:
            for step in range(num_simulations):
                new_preference = current_preference.copy()
                v = np.random.randint(n)
                i, j = np.random.choice(m, 2, replace=False)
                new_preference[i, v], new_preference[j, v] = (
                    new_preference[j, v],
                    new_preference[i, v],
                )
                new_score = reconstruct_match(new_preference)

                # Adaptive temperature based on phase
                temperature = max(
                    0.1, 1.0 - (phase * step) / (3 * num_simulations))

                # Modified acceptance probability calculation
                score_diff = new_score - current_score
                acceptance_prob = np.exp(score_diff / temperature)

                current_acceptance_rate = acceptance_rate(acceptance_window)
                if current_acceptance_rate > target_acceptance:
                    acceptance_prob *= 0.5

                accepted = score_diff > 0 or np.random.rand() < acceptance_prob
                if accepted:
                    current_preference = new_preference
                    current_score = new_score

                    if current_score > best_score:
                        best_score = current_score
                        best_preference = current_preference.copy()

                acceptance_window.append(accepted)
                if len(acceptance_window) == window_size:
                    if (
                        current_acceptance_rate <= target_acceptance
                        and current_score >= best_score * (1 - improvement_threshold)
                    ):
                        return current_preference

                    if phase == 1 and current_score == len(leaderboard):
                        phase += 1
                        break
                    elif phase == 2 and current_acceptance_rate <= target_acceptance:
                        phase += 1
                        break
                    elif phase == 3 and stability_counter >= stable_window:
                        return best_preference

                    if current_acceptance_rate <= target_acceptance:
                        stability_counter += 1
                    else:
                        stability_counter = 0

            phase += 1

        return best_preference

    def _is_valid_preference_matrix(self, matrix):
        if not isinstance(matrix, np.ndarray) or matrix.dtype.kind != "U":
            return False

        if matrix.ndim != 2:
            return False

        for col in range(matrix.shape[1]):
            if len(set(matrix[:, col])) != matrix.shape[0]:
                return False

        return True


def main():
    voter_preference = np.char.array(
        # voters:   1    2    3    4    5    6
        [
            ["B", "A", "D", "E", "E", "D", "B"],  # 1st preference
            ["C", "C", "E", "D", "D", "A", "A"],  # 2nd preference
            ["A", "E", "A", "A", "A", "C", "E"],  # 3rd preference
            ["D", "B", "C", "B", "B", "E", "D"],  # 4th preference
            ["E", "D", "B", "C", "C", "B", "C"],
        ]
    )

    print("Original preferences:")
    print(voter_preference)
    true_outome = {
        k: v
        for k, v in sorted(
            plurality_voting(voter_preference).items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    print(f"{true_outome}")

    test = ATVA3(happiness_measure=NDCG, risk_measure=FlipRewardRisk)

    reconstruct_preference, outcome, happiness, _, _, risk = test.analyze(
        voter_preference=voter_preference, voting_scheme=plurality_voting
    )
    print("\nReconstructed preferences:")
    print(reconstruct_preference)
    print(outcome)
    print("\n")
    print(happiness)
    print(risk)


if __name__ == "__main__":
    main()
