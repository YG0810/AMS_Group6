import numpy as np
import random

# all possible candidates
CANDIDATES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def generate_test(num_candidates: int, num_voters: int):
    # shuffle the slice of candidates `num_voters` times, transpose for voters on columns
    return np.char.array(
        [
            random.sample(CANDIDATES[:num_candidates], num_candidates)
            for _ in range(num_voters)
        ]
    ).T


if __name__ == "__main__":
    # where to stores test cases
    DEST = "test_cases/"

    num_candidates = [3, 5, 10]  # number of candidates
    num_voters = [5, 10, 20]  # number of voters
    num_cases = 3  # how many cases to generate for each combination
    random.seed(123)  # set seed for reproducibility

    # create test cases for all combinations
    for nc in num_candidates:
        for nv in num_voters:
            for i in range(num_cases):
                votes = generate_test(nc, nv)
                np.save(DEST + f"{nc}c_{nv}v_{i}.npy", votes)
