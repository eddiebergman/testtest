from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import math

HERE = Path(__file__).parent.resolve()

# Paths to the test data
paths = {
    "MC": HERE / "MCTestData.csv",
    "TMSt": HERE / "TMStTestData.csv",
    "F": HERE / "FTestData.csv",
}


def load_data_MNTest() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data stored in McNemarTest.csv
    :param fl: path to the csv file
    :return: labels, prediction1, prediction2
    """
    data = pd.read_csv(paths["MC"], header=None).to_numpy()
    labels = data[:, 0]
    prediction_1 = data[:, 1]
    prediction_2 = data[:, 2]
    return labels, prediction_1, prediction_2


def load_data_TMStTest() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads data stored in fl
    :param fl: path to the csv file
    :return: y1, y2
    """
    data = np.loadtxt(paths["TMSt"], delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def load_data_FTest() -> np.ndarray:
    """
    Loads data stored in fl
    :param fl: path to the csv file
    :return: evaluations
    """
    errors = np.loadtxt(paths["F"], delimiter=",")
    return errors


def McNemar_test(
    labels: np.ndarray,
    prediction_1: np.ndarray,
    prediction_2: np.ndarray,
) -> float:
    """
    TODO
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    B = np.count_nonzero(np.where((prediction_1 == labels) & (prediction_1 != prediction_2), 1, 0))
    C = np.count_nonzero(np.where((prediction_2 == labels) & (prediction_1 != prediction_2), 1, 0))
    chi2_Mc = (abs(B - C) - 1) ** 2 / (B + C)
    return chi2_Mc


def TwoMatchedSamplest_test(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    TODO
    :param y1: runs of algorithm 1
    :param y2: runs of algorithm 2
    :return: the test statistic t-value
    """
    d_line = np.sum(y1 - y2) / y1.size
    inner_sum = np.sum((y1 - y2 - d_line) ** 2)
    sigma_d1 = math.sqrt(1 / (y1.size - 1) * inner_sum)
    t_value = math.sqrt(y1.size) * d_line / sigma_d1
    return t_value


def Friedman_test(errors: np.ndarray) -> Tuple[float, dict]:
    """
    TODO
    :param errors: the error values of different algorithms on different datasets
    :return: chi2_F: the test statistic chi2_F value
    :return: FData_stats: the statistical data of the Friedan test data, containing anything
    you need to solve the `Nemenyi_test` and `box_plot` functions.
    """
    ranks = []
    for input in errors:
        output = [0] * len(input)
        for i, x in enumerate(sorted(range(len(input)), key=lambda y: input[y])):
            output[x] = i + 1
        ranks.append(output)
    ranks_np = np.array(ranks)

    # Now we compute the chi2_F by applying the formulas
    n = errors.shape[0]
    k = errors.shape[1]
    r_j_bar = np.average(ranks_np, axis=0)
    r_bar = np.average(ranks_np)
    ss_total = n * np.sum((r_j_bar - r_bar) ** 2)
    ss_error = 1 / (n * (k - 1)) * np.sum((ranks_np - r_bar) ** 2)
    chi2_F = ss_total / ss_error
    FData_stats = {
        'errors': errors,
        'avg_ranks': r_j_bar
    }
    return chi2_F, FData_stats


def Nemenyi_test(fdata_stats: dict) -> np.ndarray:
    """
    TODO
    :param fdata_stats np.ndarray: the statistical data of the Friedan test data to be utilized in the post hoc problems
    :return: the test statisic Q value
    """
    alg_count = fdata_stats['errors'].shape[1]
    datasets_count = fdata_stats['errors'].shape[0]
    Q_value = np.zeros((alg_count, alg_count))
    avg_ranks = fdata_stats['avg_ranks']

    for i in range(alg_count):
        for j in range(alg_count):
            if j > i:
                Q_value[i][j] = (avg_ranks[i] - avg_ranks[j]) / \
                    (math.sqrt((alg_count * (alg_count + 1)) / (6 * datasets_count)))
    return Q_value


def box_plot(fdata_stats: dict) -> None:
    """
    TODO
    :param fdata_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    """
    pass


def main() -> None:
    # (a)
    labels, prediction_A, prediction_B = load_data_MNTest()
    chi2_Mc = McNemar_test(labels, prediction_A, prediction_B)
    print("chi2_Mc ", chi2_Mc)

    # (b)
    y1, y2 = load_data_TMStTest()
    t_value = TwoMatchedSamplest_test(y1, y2)
    print("t_value ", t_value)

    # (c)
    errors = load_data_FTest()
    chi2_F, FData_stats = Friedman_test(errors)
    print("chi2_F ", chi2_F)

    # (d)
    Q_value = Nemenyi_test(FData_stats)
    print("Q value ", Q_value)

    # (e)
    box_plot(FData_stats)


if __name__ == "__main__":
    main()
