import numpy as np
import pandas as pd
import pytest

from experiment import ExperimentEvaluator


@pytest.mark.skip("Not for now")
def test_check_extra_points_raises_error():
    sprints, stories = [
        pd.read_csv(f"data/{name}.csv") for name in ("4sprints", "4stories")
    ]
    test_list = [
        [3, 0, 0, 0],  # stories.loc[22, 'estimated_size']  2 points
        [0, 4, 0, 0],  # stories.loc[98, 'estimated_size']  3 points
        [0, 0, 4, 0],  # stories.loc[200, 'estimated_size']  3 points
        [0, 0, 0, 4],  # stories.loc[286, 'estimated_size']  3 points
    ]

    for extra_points_list in test_list:
        algorithm_outcome = pd.DataFrame(
            {
                "story_id": [21, 97, 199, 285],
                "extra_points": extra_points_list,
            }
        )
        ev = ExperimentEvaluator(algorithm_outcome, sprints, stories)
        with pytest.raises(AssertionError):
            ev.run()


def test_compute_diff_updates_outcome_with_the_right_values():
    sprints, stories, algorithm_outcome = [
        pd.read_csv(f"data/{name}.csv")
        for name in ("4sprints", "4stories", "algorithm_outcome")
    ]
    ev = ExperimentEvaluator(algorithm_outcome, sprints, stories)
    ev._compute_diff()
    assert ev.outcome.columns.tolist() == [
        "story_id",
        "extra_points",
        "estimated_size",
        "real_size",
        "total_real_size",
        "real_length",
        "point_distance",
        "distance",
    ]
    assert ev.outcome.shape == (4, 8)
    assert np.all(ev.outcome.story_id.values == algorithm_outcome.story_id.values)
    assert np.all(ev.outcome.extra_points == algorithm_outcome.extra_points)

    estimated_size = np.array(
        [
            49,  # stories.loc[0:21, "estimated_size"].sum()  # visually check the numbers
            32,  # stories.loc[91:97, "estimated_size"].sum()
            46,  # stories.loc[182:199, "estimated_size"].sum()
            36,  # stories.loc[273:285, "estimated_size"].sum()
        ]
    )
    assert np.all(ev.outcome.estimated_size.tolist() == estimated_size)

    real_size = np.array(
        [
            58,  # stories.loc[0:21, "real_size"].sum()
            31,  # stories.loc[91:97, "real_size"].sum()
            62,  # stories.loc[182:199, "real_size"].sum()
            46,  # stories.loc[273:285, "real_size"].sum()
        ]
    )
    assert np.all(ev.outcome.real_size == real_size)

    total_real_size = real_size + algorithm_outcome.extra_points
    assert np.all(ev.outcome.total_real_size == total_real_size)
    assert np.all(ev.outcome.real_length == sprints.real_length)
    assert np.all(ev.outcome.point_distance == sprints.real_length - total_real_size)
    assert np.all(
        ev.outcome.distance == ev.outcome.point_distance / stories.real_size.mean()
    )

    # visually check the numbers
    assert np.all(total_real_size == np.array([58, 31, 64, 46]))
    assert np.all(ev.outcome.point_distance == np.array([4, 3, -2, -4]))
    assert np.allclose(
        ev.outcome.distance, np.array([1.157393, 0.868045, -0.578696, -1.157393])
    )


def test_compute_regret_updates_outcome_with_the_right_values():
    sprints, stories, algorithm_outcome = [
        pd.read_csv(f"data/{name}.csv")
        for name in ("4sprints", "4stories", "algorithm_outcome")
    ]
    ev = ExperimentEvaluator(algorithm_outcome, sprints, stories)
    ev._compute_diff()
    ev._compute_regret()
    expected_regret = np.array(
        [1.157393, 0.868045**2, (-0.578696) ** 2, (-1.157393) ** 2]
    )
    assert np.allclose(ev.outcome.regret, expected_regret)


def test_run_evaluator_returns_the_right_value():
    names = ("4sprints", "4stories", "algorithm_outcome")
    sprints, stories, algorithm_outcome = [
        pd.read_csv(f"data/{name}.csv") for name in names
    ]
    ev = ExperimentEvaluator(algorithm_outcome, sprints, stories)
    expected_regret = np.array(
        [1.157393, 0.868045**2, (-0.578696) ** 2, (-1.157393) ** 2]
    ).sum()
    ev.run()
    total_regret = ev.outcome.regret.sum()
    assert abs(total_regret - expected_regret) < 0.00001
