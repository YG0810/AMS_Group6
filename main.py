import builtins
import numpy as np
from voting_methods import *

# get rid of `np.str_` in the print output
oprint = builtins.print
def cprint(*args, **kwargs):
    oprint(*[{str(k): v for k, v in a.items()} if isinstance(a, dict) else a for a in args], **kwargs)
builtins.print = cprint

voter_preference = np.array(
# voters:   1    2    3    4
         [["B", "A", "C", "C"], # preference 1 (best)
          ["C", "C", "B", "B"], # preference 2
          ["A", "B", "A", "A"]] # preference 3 (worst)
)

result = two_person_voting(voter_preference)
print(result)
