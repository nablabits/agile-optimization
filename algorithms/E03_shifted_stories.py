import numpy as np
import pandas as pd

from algorithms import BaseAlgorithm


HS = np.arange(1, 90)


class ShiftedStoryPoints:
    """Slightly shift the story points to account for uncertainty."""

    story_points = np.array([1, 2, 3, 5, 8, 13, 20])

    def __init__(self, lower_value_for_one=0.5):
        self.lower_value_for_one = lower_value_for_one
        self.bins = self.story_points.copy()
        self.boundaries = None
        self.boundaries_h = None

    def _refine_bins(self):
        increasing_factor = self.bins * np.arange(self.bins.size)
        increasing_factor = increasing_factor / increasing_factor.sum()
        return self.bins * (1 + increasing_factor)

    def run(self):
        upper = self._refine_bins()
        lower = np.append([self.lower_value_for_one], upper[:-1])
        self.boundaries = pd.DataFrame(
            {"lower": lower, "upper": upper}, index=self.story_points
        )

        # Basically, above thresholds set a range of uncertainty. For instance, 1 point is
        # a story that takes something between .5 points and 1 point. This assumes
        # implicitly that we tend to overestimate the stories. However, we rather tend to
        # underestimate the stories, so we might want to place the story points in the
        # middle of the range instead of at the end of it.
        shift = (self.boundaries.upper - self.boundaries.lower) / 2
        self.boundaries["shift_points"] = self.boundaries.index + shift.round().astype(
            int
        )
        return self.boundaries


class E03ShiftedStories(BaseAlgorithm):
    """
    Shift up the estimated size of the stories proportionally to their size given that we tend to
    underestimate them and then use the simple algorithm recalling mean approach to compute the
    estimation for the sprint.

    Just by applying the shift to the stories, there's an important improvement, the distance
    between estimated and real size gets reduced from -314672 to 22426, where the negative
    represents underestimation and the positive overestimation.
    """

    def __init__(self, sprints, stories, compute_size=None):
        super().__init__(sprints, stories, compute_size=compute_size)

        # The intuition of this algorithm is that we will shift the stories by some amount, so
        # let's do it.
        self.shift_stories()

    def shift_stories(self):
        boundaries = ShiftedStoryPoints().run()
        shift_points = pd.merge(
            self.stories,
            boundaries.shift_points,
            how="left",
            left_on="estimated_size",
            right_index=True,
        ).shift_points.values
        self.stories.loc[:, "estimated_size"] = shift_points

    def estimate_length(self):
        """Override the main method."""
        if self.current_sprint < 1:
            # for the first sprint we use the current points (it means 1pt per day and per person)
            return self.sprints.at[self.current_sprint, "points"]

        # otherwise, we return the mean of the historical real lengths. Recall that sprint.points
        # are also persons x days.
        history = self.sprints[self.sprints.index < self.current_sprint]
        mean_ratio = history.real_length.sum() / history.points.sum()
        return (self.sprints.at[self.current_sprint, "points"] * mean_ratio).astype(int)
