import numpy as np
from BTVA import BTVA
from happiness_measure import NDCG
from voting_schemes import plurality_voting

voter_preference = np.char.array(
    # voters:   1    2    3    4
    [
        ["B", "A", "C", "C"],  # 1st preference
        ["C", "C", "B", "B"],  # 2nd preference
        ["A", "B", "A", "A"],
    ]  # 3rd preference
)

# Analyze the voting preferences
btva = BTVA(happiness_measure=NDCG)
outcome, happiness, overall_happiness, _, _ = btva.analyze(
    voter_preference, plurality_voting
)
print(outcome)
print(happiness)

# Sanity checking overall happiness
assert sum(happiness) == overall_happiness

print(overall_happiness)
