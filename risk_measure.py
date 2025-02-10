import numpy as np
from typing import List

def flip_reward_risk(voter_preference: np.ndarray,
                                    individual_happiness: List[float],
                                    strategic_options: list,
                                    p: float = 1.7) -> tuple:
    """
    Computes the likelihood of strategic voting by evaluating the trade-off between preference changes 
    and potential happiness gains. For each voter, analyzes how likely they are to deviate from their 
    true preferences when presented with strategic options that could increase their happiness.

    The risk score (between 0 and 1) reflects how tempting a strategic option is, based on:
    1. How much happiness could be gained (delta happiness)
    2. How many preference switches are needed (inversion distance)
    Higher scores suggest higher likelihood of strategic voting.

    Args:
        voter_preference (np.ndarray): Original preference rankings for each voter
        individual_happiness (List[float]): Current happiness values for each voter (must be floats between 0 and 1, meaning normalized)
        strategic_options (list): List of possible strategic options for each voter, 
                                where each option is a tuple (preference_ranking, expected_happiness)
        p (float): Sensitivity parameter controlling risk assessment, must be in range [1.3, 1.7]
        - p=1.3 : Conservative assessment
            Assigns high risk mainly when large happiness gains require few preference changes
        - p=1.7 : Stringent assessment (preferably)
            More likely to flag subtle strategic opportunities
            Assumes voters are tempted even by small happiness gains if few changes needed

    Returns:
        tuple: (risks, overall_max_risk) where:
            - risks: List of (preference, risk_score) tuples for each voter's highest risk option
            - overall_max_risk: Maximum risk score across all voters (float between 0 and 1)
    
    Raises:
        ValueError: If p is not in the range [1.3, 1.7]
    """

    if not 1.3 <= p <= 1.7:
        raise ValueError(f"Parameter p must be in range [1.3, 1.7], got {p}")
   
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

""" Example Usage """

voter_preference = np.array([
    ['A', 'B', 'C', 'D'],  # Voter 1's original preference
    ['B', 'A', 'D', 'C'],  # Voter 2's original preference
    ['C', 'D', 'A', 'B']   # Voter 3's original preference
])

individual_happiness = [0.4, 0.3, 0.6]

strategic_options = [
    [  # Options for Voter 1
        (['B', 'A', 'C', 'D'], 0.8),   # Option 1
        (['C', 'A', 'B', 'D'], 0.6)    # Option 2
    ],
    [  # Options for Voter 2
        (['A', 'B', 'D', 'C'], 0.8)    # Option 1
    ],
    [  # Options for Voter 3
        (['D', 'A', 'C', 'B'], 0.79),   # Option 1
        (['D', 'B', 'C', 'A'], 0.65)    # Option 2
    ]
]

p = 1.7 # logically relevant p in [1.3, 1.6]

individual_risks, overall_max_risk = flip_reward_risk(
    voter_preference,  
    individual_happiness, 
    strategic_options, 
    p
)

for i, (voter_option, risk) in enumerate(zip(strategic_options, individual_risks), 1):
    print(f"\nVoter {i}:")
    print(f"Original Preference: {voter_preference[i-1]}")
    print("Strategic Options:")
    for option in voter_option:
        inv_dist = inversion_ranking_distance(voter_preference[i-1], option[0])
        print(f"  - Preference: {option[0]}, "
              f"Difference in Happiness: {round(abs(individual_happiness[i-1] - option[1]), ndigits=3)}, "
              f"Inversion Distance: {inv_dist:.4f}")
    print(f"Best Strategic Option: {risk[0]} (Risk Score: {risk[1]:.4f})")

print(f"\nOverall Maximum Risk: {overall_max_risk:.4f}")