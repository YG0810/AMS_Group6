from dataclasses import dataclass
from multiprocessing.pool import Pool
import pickle
from time import time
import numpy as np
from numpy.typing import NDArray
from os import listdir
from ATVA4 import ATVA4
from BTVA import BTVA, HappinessMeasure, RiskMeasure, VotingScheme
import happiness_measure
from pandas import DataFrame
from happiness_measure import get_happiness
from risk_measure import FlipRewardRisk, JointFlipRewardRisk, WinnerChangeRisk, probStrategicVoting
from strategy_generators import StrategyGenerator
import strategy_generators
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


@dataclass
class RiskMeasureEntry:
    name: str
    risk_values: list[float]


@dataclass
class HappinessMeasureEntry:
    name: str
    happiness_values: list[float]
    risk_measures: list[RiskMeasureEntry]


@dataclass
class VotingSchemeEntry:
    name: str
    happiness_measures: list[HappinessMeasureEntry]


@dataclass
class TestOutputEntry:
    input: np.chararray
    voting_schemes: list[VotingSchemeEntry]


result = TestOutputEntry(np.char.array("Test"), [])


for votingSchemeEntry in result.voting_schemes:
    print(f"Voting Scheme: {votingSchemeEntry.name}")

    for happinessMeasureEntry in votingSchemeEntry.happiness_measures:
        print(f"\tHappiness measure: {happinessMeasureEntry.name}")
        print(f"\tHappiness values: {happinessMeasureEntry.happiness_values}")

        for riskMeasureEntry in happinessMeasureEntry.risk_measures:
            print(f"\tRisk measure: {riskMeasureEntry.name}")
            print(f"\tRisk values: {riskMeasureEntry.risk_values}")


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


def testMatrix() -> DataFrame:
    df = DataFrame(
        columns=[
            "input",
            "voting_scheme",
            "happiness_measure",
            "happiness_values",
            "risk_measure",
            "risk_values",
        ]
    )
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
        NamedRiskMeasure("Winner change risk", WinnerChangeRisk),
        NamedRiskMeasure("Flip Reward", FlipRewardRisk),
        # NamedRiskMeasure("Joing Flip Reward", JointFlipRewardRisk)
        #   Yannick mentioned that this is primarily for ATVA, so this is disabled.
        #   However, this serves as a reminder to implement once ATVAs work.
        NamedRiskMeasure("Probability of Strategic Voting",
                         probStrategicVoting),
    ]


    for vs in votingSchemes:
        for hm in happiness_measures:
            for rm in risk_measures:
                btva = ATVA4(hm.happiness_measure,
                            rm.risk_measure,
                            strategy_generators.createNDistinctPermutations)

                for tc in testScenarios:
                    startTime = time()
                    print("---")
                    print(f"Testing {tc.shape}")
                    print(f"\tVoting scheme: {vs.label}")
                    print(f"\tHappiness measure: {hm.label}")
                    print(f"\tRisk measure: {rm.label}")
                    tcCA = np.char.array(tc)
                    result = btva.analyze(tcCA, vs.votingScheme)

                    df.loc[len(df)] = {  # type:ignore
                        "input": tc,
                        "voting_scheme": vs.label,
                        "happiness_measure": hm.label,
                        "happiness_values": result[1],
                        "risk_measure": rm.label,
                        "risk_values": result[-1],
                    }

                    endTime = time()

                    print(f"Time taken: {endTime-startTime}s")
    return df


def main():
    df = testMatrix()

    df.to_csv("testOutputATVA4.csv", index=False)

    with open("testOutputATVA4.pkl", "wb") as file:
        # Serialize the object and write it to the file
        pickle.dump(df, file)


if __name__ == "__main__":
    main()
