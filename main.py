import numpy as np


def anti_plurality_voting(voter_preference: np.ndarray) -> dict:
    """
    anti-plurality voting

    :param voter_preference: A numpy array of shape (m,n), where m is the number of candidates preference, and n is the number of voters.

    :return: A dictionary with the candidates and their corresponding value by anti-plurality voting.
    {
        candidate_A: value_candidate_A,
        candidate_B: value_candidate_B,
        ...
    }
    """
    m, n = voter_preference.shape
    candidates = np.unique(voter_preference)
    candidate_values = {candidate: 0 for candidate in candidates}

    # Assign values to candidates
    for preference in range(m):
        if preference == m - 1:
            break
        for voter in range(n):
            candidate_values[voter_preference[preference, voter]] += 1

    return candidate_values


""" Example

Matrix of voter preferences

      Voters: 1   2   3   4
Preference 1: B | A | C | C
Preference 2: C | C | B | B
Preference 3: A | B | A | A
"""

voter_preference = np.array(
    [["B", "A", "C", "C"], ["C", "C", "B", "B"], ["A", "B", "A", "A"]]
)

result = anti_plurality_voting(voter_preference)
print(result)
