import numpy as np
from BTVA import BTVA
from voting_schemes import *

voter_preference = np.char.array(
# voters:   1    2    3    4
         [['B', 'A', 'C', 'C'], # 1st preference
          ['C', 'C', 'B', 'B'], # 2nd preference
          ['A', 'B', 'A', 'A']] # 3rd preference
)

# Create BTVA (TODO: happiness and risk measures)
btva = BTVA(happiness_measure=lambda x,y: 1, risk_measure=lambda x,y,z: 1)

# Analyze the voting preferences
outcome, _, _, _, _ = btva.analyze(voter_preference, plurality_voting)
print(outcome)
