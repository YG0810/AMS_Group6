import numpy as np
from numpy.char import chararray as npchar

class BTVA:

    def __init__(self, happiness_measure: function, risk_measure: function):
        self.happiness_measure = happiness_measure
        self.risk_measure = risk_measure

    def analyze(self, voter_preference: npchar, voting_scheme: function) -> tuple:
        """
        Analyze the voting preference of a group of voters using a specific voting scheme.

        :param voter_preference: An array of shape (m,n), where m is the number of candidates and n is the number of voters.
        :param voting_scheme: The voting scheme to use.
        :return: a tuple containing the following:
                 - Non-strategic voting outcome
                 - The happiness level of each voter
                 - The overall happiness level
                 - For each voter a (possibly empty) set of strategic-voting options
                 - Overall risk of strategic voting for the given input
        """
        m, n = voter_preference.shape

        # Non-strategic voting outcome
        outcome = voting_scheme(voter_preference)

        # Happiness levels
        individual_happiness = [self.happiness_measure(voter_preference[:, i], outcome) for i in range(n)]
        overall_happiness = sum(individual_happiness)

        # Strategic voting options
        strategic_options = [] # TODO
        risk = self.risk_measure(voter_preference, voting_scheme, strategic_options)

        return outcome, individual_happiness, overall_happiness, strategic_options, risk
