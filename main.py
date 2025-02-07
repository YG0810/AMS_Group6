import numpy as np
from BTVA import BTVA
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
btva = BTVA()
outcome, _, _, _, _ = btva.analyze(voter_preference, plurality_voting)
print(outcome)
