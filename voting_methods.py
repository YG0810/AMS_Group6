import numpy as np

def anti_plurality_voting(voter_preference: np.ndarray) -> dict:
    """
    Implements anti-plurality voting where candidates receive points based on their ranking position.

    :param voter_preference: A numpy array of shape (m,n), where m is the number of candidates preference and n is the number of voters.

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

def borda_count_voting(voter_preference: np.ndarray) -> dict:
    """
    Implements Borda count voting where candidates receive points based on their ranking position.
    The candidate ranked first gets m-1 points, second gets m-2 points, etc., last gets 0 points.
    
    :param voter_preference: A numpy array of shape (m,n), where m is the number of preference ranks,
                           and n is the number of voters. Each element contains the candidate name/id.
    :return: A dictionary with candidates and their corresponding Borda scores
    """
    m, n = voter_preference.shape
    candidates = np.unique(voter_preference)
    candidate_values = {candidate: 0 for candidate in candidates}
    
    # For each voter, assign points based on preference rank
    for preference_rank in range(m):
        points = m - 1 - preference_rank  # First gets m-1 points, second m-2, etc.
        for voter in range(n):
            candidate_values[voter_preference[preference_rank, voter]] += points
            
    return candidate_values

"""
Example

Matrix of voter preferences

      Voters: 1   2   3   4
Preference 1: B | A | C | C
Preference 2: C | C | B | B
Preference 3: A | B | A | A
"""

def two_person_voting(voter_preference: np.ndarray) -> dict:
    """
    Implements two-person voting where each voter selects two candidates.
    
    :param voter_preference: A numpy array of shape (m,n), where m is the number of candidates preference and n is the number of voters.
    :return: A dictionary with the candidates and their corresponding value by two-person voting.
    """
    m, n = voter_preference.shape
    candidates = np.unique(voter_preference)
    candidate_values = {candidate: 0 for candidate in candidates}

    # Assign values to candidates
    for preference in range(m):
        for voter in range(n):
            candidate_values[voter_preference[preference, voter]] += 1
        if preference == 1: # only count the first two preferences
            break
    return candidate_values
