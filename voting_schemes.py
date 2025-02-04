import numpy as np
from numpy.char import chararray as npchar

def anti_plurality_voting(voter_preference: npchar) -> dict:
    """
    Implements anti-plurality voting: each voter votes against one candidate, the candidate with the fewest votes wins.

    :param voter_preference: An array of shape (m,n), where m is the number of candidates and n is the number of voters.
    :return: A dictionary with the candidates and their corresponding value by anti-plurality voting (m - #votes).
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

def borda_count_voting(voter_preference: npchar) -> dict:
    """
    Implements Borda count voting: candidate ranked first gets m-1 points, second gets m-2 points, etc., last gets 0 points.

    :param voter_preference: An array of shape (m,n), where m is the number of candidates and n is the number of voters.
    :return: A dictionary with the candidates and their corresponding Borda scores.
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

def two_person_voting(voter_preference: npchar) -> dict:
    """
    Implements two-person voting: each voter votes for their top two candidates.

    :param voter_preference: An array of shape (m,n), where m is the number of candidates and n is the number of voters.
    :return: A dictionary with the candidates and their corresponding value by two-person voting.
    """
    m, n = voter_preference.shape
    candidates = np.unique(voter_preference)
    candidate_values = {candidate: 0 for candidate in candidates}

    # Assign values to candidates
    for preference in range(m):
        for voter in range(n):
            candidate_values[voter_preference[preference, voter]] += 1
        if preference == 1: # Only count the top two preferences
            break
    return candidate_values

def plurality_voting(voter_preference: npchar) -> dict:
    """
    Implements plurality voting: each voter votes for their top candidate.

    :param voter_preference: An array of shape (m,n), where m is the number of candidates and n is the number of voters.
    :return: A dictionary with the candidates and their corresponding value by plurality voting.
    """
    _, n = voter_preference.shape
    candidates = np.unique(voter_preference)
    candidate_values = {candidate: 0 for candidate in candidates}

    # Assign values to candidates
    for voter in range(n):
        candidate_values[voter_preference[0, voter]] += 1
    return candidate_values
