""" Hapiness Measurements """

def get_happiness(voter_preference, voting_outcome, preference_weights: list = None, distance_weights: list = None):

    # Set default weights
    if not preference_weights:
        preference_weights = [1] * len(voter_preference)
    if not distance_weights:
        distance_weights = [1] * len(voter_preference)

    # Measure happiness
    happiness_score = 0
    for preference_idx in range(len(voter_preference)):
        # Retrieve distance between preference and outcome
        distance = abs(preference_idx - voting_outcome.index(voter_preference[preference_idx]))

        # Calculate happiness score
        happiness_score += preference_weights[preference_idx] * distance_weights[distance]

    return happiness_score
    

""" Example Usage """

# Top-1 Binary happiness measure
preference_weights = [1, 0, 0]
distance_weights = [1, 0, 0]

print(get_happiness(['A', 'B', 'C'], ['A', 'B', 'C'], preference_weights, distance_weights))

# Bottom-1 Binary happiness measure
preference_weights = [1, 0, 0]
distance_weights = [0, 0, -1]

print(get_happiness(['A', 'B', 'C'], ['C', 'B', 'A'], preference_weights, distance_weights))
