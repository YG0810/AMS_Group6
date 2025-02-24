# This file contains common types used in the code base
# And avoids redefining existing types and causing confusion

from typing import Any
from collections.abc import Callable
import numpy as np

VoterPreference = np.char.chararray

CandidateResults = dict[str, int]

VotingScheme = Callable[[VoterPreference], CandidateResults]

# (voter_preference, outcome, weights) -> float
HappinessMeasure = Callable[
    [VoterPreference, list[str], list[float] | None, list[float] | None], float
]
# (voter_preference,voting_scheme, individual_happiness, strategic_options) -> float
RiskMeasure = Callable[[VoterPreference,
                        VotingScheme, list[float], list[Any]], float]