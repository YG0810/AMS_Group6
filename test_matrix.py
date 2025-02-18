from dataclasses import dataclass
import pickle
import numpy as np
from numpy.typing import NDArray
from os import listdir
from BTVA import BTVA, HappinessMeasure, RiskMeasure, VotingScheme
import BTVALite
import happiness_measure
from measurements import get_happiness
from risk_measure import FlipRewardRisk, JointFlipRewardRisk, probStrategicVoting
from voting_schemes import (
    anti_plurality_voting,
    borda_count_voting,
    plurality_voting,
    two_person_voting,
)


@dataclass
class NamedVotingScheme:
    label: str
    votingScheme: VotingScheme


@dataclass
class NamedHappinessMeasure:
    label: str
    happiness_measure: HappinessMeasure


@dataclass
class NamedRiskMeasure:
    label: str
    risk_measure: RiskMeasure


# List of (voterPreferences, VotingScheme, HappinessMeasure, RiskMeasure, IndividualHappiness, overallHappiness)
TestMatrixOutput = list[
    tuple[
        np.chararray,
        NamedVotingScheme,
        NamedHappinessMeasure,
        NamedRiskMeasure,
        list[float],
        float,
    ]
]


def loadData() -> list[NDArray[np.str_]]:
    files = listdir("test_cases")

    test_scenarios: list[NDArray[np.str_]] = []
    for file in files:
        try:
            scenario = np.load(f"test_cases/{file}")
            test_scenarios.append(scenario)
            print(f"test_cases/{file}:", scenario.shape)
        except Exception as e:
            print(f"Error reading {file} as test_case ({e})")

    return list(reversed(test_scenarios))


def testMatrix() -> TestMatrixOutput:
    testScenarios = loadData()
    votingSchemes = [
        NamedVotingScheme("Anti-plurality voting", anti_plurality_voting),
        NamedVotingScheme("Borda count voting", borda_count_voting),
        NamedVotingScheme("Two person voting", two_person_voting),
        NamedVotingScheme("Plurality voting", plurality_voting),
    ]

    happiness_measures = [
        NamedHappinessMeasure("NDCG", happiness_measure.NDCG),
        NamedHappinessMeasure("Kendall Tau", happiness_measure.KendallTau),
        NamedHappinessMeasure(
            "Bubble sort distance / Kendall Tau Distance",
            happiness_measure.BubbleSortDistance,
        ),
        NamedHappinessMeasure("Get Happiness", get_happiness),
    ]

    risk_measures = [
        NamedRiskMeasure("Flip Reward", FlipRewardRisk),
        # NamedRiskMeasure("Joing Flip Reward", JointFlipRewardRisk)
        #   Yannick mentioned that this is primarily for ATVA, so this is disabled.
        #   However, this serves as a reminder to implement once ATVAs work.
        NamedRiskMeasure("Probability of Strategic Voting", probStrategicVoting),
    ]

    outputs: TestMatrixOutput = []

    for vs in votingSchemes:
        for hm in happiness_measures:
            for rm in risk_measures:
                btva = BTVALite.BTVALite(hm.happiness_measure, rm.risk_measure)

                for tc in testScenarios:
                    tcCA = np.char.array(tc)
                    result = btva.analyze(tcCA, vs.votingScheme)

                    entry = (tcCA, vs, hm, rm, result[0], result[1])
                    outputs.append(entry)
                    print(entry)

    return outputs


def main():
    outputs = testMatrix()

    with open("testOutput.pkl", "wb") as file:
        # Serialize the object and write it to the file
        pickle.dump(outputs, file)


if __name__ == "__main__":
    main()
