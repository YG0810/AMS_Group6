import numpy as np
from voting_schemes import *

voter_preference = np.char.array(
# voters:   1    2    3    4
         [['B', 'A', 'C', 'C'], # 1st preference
          ['C', 'C', 'B', 'B'], # 2nd preference
          ['A', 'B', 'A', 'A']] # 3rd preference
)

result = plurality_voting(voter_preference)
print(result)
