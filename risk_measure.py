import numpy as np
from typing import List

def flip_reward_risk(voter_preference: np.ndarray,
                                    individual_happiness: List[float],
                                    strategic_options: list,
                                    p: float) -> tuple:
    """
    Computes the risk of strategic dishonesty in voting based on preference changes and happiness deltas.
    The risk score combines the happiness difference and preference rearrangement distance, normalized
    between 0 and 1. Higher scores indicate higher risk of strategic manipulation.
    
    Args:
        voter_preference (np.ndarray): Original preference rankings for each voter
        individual_happiness (List[float]): Current happiness values for each voter (must be floats between 0 and 1, meaning normalized)
        strategic_options (list): List of possible strategic options for each voter, 
                                where each option is a tuple (preference_ranking, expected_happiness)
        p (float): Power parameter controlling the risk calculation sensitivity, recommended values in range [1.3, 1.6]
        
    Returns:
        tuple: (risks, overall_max_risk) where:
            - risks: List of (preference, risk_score) tuples for each voter's highest risk option
            - overall_max_risk: Maximum risk score across all voters (float between 0 and 1)
    """

    risks = []
    for i, options in enumerate(strategic_options):
        if not options:
            risks.append(set())
            continue
        
        risks4i = []
        for pref in options:
            preference, happiness = pref
            norm_dist = inversion_ranking_distance(voter_preference[i], preference)
            delta_happ = abs(happiness - individual_happiness[i])
            score = np.tanh(delta_happ / np.log(norm_dist**(p-1) + 1))
            risks4i.append((preference, score))
        
        max_risk_option = max(risks4i, key=lambda x: x[1])
        risks.append(max_risk_option)
    overall_max_risk = max(risks, key=lambda x: x[1] if isinstance(x, tuple) else 0.0)[1]
    return risks, overall_max_risk

def inversion_ranking_distance(base_pref: list, option_pref: list) -> float:
    """
    Computes the normalized minimal rearrangement distance between two rankings based on the number of
    displacements required to transform one ranking into another. The result is normalized between 0 and 1,
    where 0 means the rankings are identical and 1 means they are completely reversed.
    
    Args:
        base_pref (list): The reference ranking (initial preference of a candidate)
        option_pref (list): The ranking to be rearranged to match the base ranking
        
    Returns:
        float: Normalized minimal rearrangement distance between 0 and 1
        
    Raises:
        ValueError: If rankings have different lengths
    """
    if len(base_pref) != len(option_pref):
        raise ValueError("Rankings must have equal length")
        
    target_positions = {value: i for i, value in enumerate(base_pref)}
    current_positions = [target_positions[value] for value in option_pref]

    inversions = 0
    n = len(current_positions)
    for i in range(n):
        for j in range(i + 1, n):
            if current_positions[i] > current_positions[j]:
                inversions += 1

    max_inversions = (n * (n - 1)) // 2
    return inversions / max_inversions if max_inversions > 0 else 0.0

def main():
    voter_preference = np.array([
        ['A', 'B', 'C'],  # Voter 1's original preference
        ['B', 'A', 'C'],  # Voter 2's original preference
        ['C', 'A', 'B']   # Voter 3's original preference
    ])

    individual_happiness = [0.7, 0.5, 0.6]

    strategic_options = [
        [  # Options for Voter 1
            (['B', 'A', 'C'], 0.8),   # Option 1
            (['C', 'A', 'B'], 0.75)    # Option 2
        ],
        [  # Options for Voter 2
            (['A', 'B', 'C'], 0.9)    # Option 1
        ],
        [  # Options for Voter 3
            (['A', 'C', 'B'], 0.9),   # Option 1
            (['B', 'C', 'A'], 0.65)    # Option 2
        ]
    ]

    p = 1.5 # logically relevant p in [1.3, 1.6]

    individual_risks, overall_max_risk = flip_reward_risk(
        voter_preference,  
        individual_happiness, 
        strategic_options, 
        p
    )

    print("Voter Strategic Options and Risks:")
    for i, (voter_option, risk) in enumerate(zip(strategic_options, individual_risks), 1):
        print(f"\nVoter {i}:")
        print(f"Original Preference: {voter_preference[i-1]}")
        print("Strategic Options:")
        for option in voter_option:
            print(f"  - Preference: {option[0]}, Expected Happiness: {option[1]}")
        print(f"Best Strategic Option: {risk[0]} (Risk Score: {risk[1]:.4f})")

    print(f"\nOverall Maximum Risk: {overall_max_risk:.4f}")

if __name__ == "__main__":
    main()