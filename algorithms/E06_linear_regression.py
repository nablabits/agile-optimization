import pandas as pd
from scipy.stats import linregress

from algorithms import BaseAlgorithm


class E06LinearRegression(BaseAlgorithm):
    """
    This algorithm is based in the assumption that we can compute a linear model on previous
    sprints to come up with a slope and an intercept to shift both sprints' estimated length and
    stories' estimated sizes
    """

    def __init__(self, sprints, stories, compute_size=None, dynamic_update=False):
        super().__init__(sprints, stories, compute_size=compute_size)

        # We need to keep the original set of stories and sprints to calculate the regression
        # coefficient as self.sprints and self.stories will be updated with the values of the
        # regressors. At some point I ran the regression over the new updated values but it turned
        # out not a good idea.
        self.original_stories = stories.copy()
        self.original_sprints = sprints.copy()

        # track a few data of the regressors
        self.regressor_tape = pd.DataFrame(columns=[
            "stories_slope", "stories_intercept", "sprint_slope", "sprint_intercept"
        ])

    def estimate_length(self):
        if self.current_sprint < 2:
            # for the first two sprints we use the current points (it means 1pt per day and per
            # person)
            return self.sprints.at[self.current_sprint, "points"]

        # otherwise, we want to update the sizes and the length for the current
        # sprint before doing any calculation.
        self.update_sprint_values()

        # then we just return the new calculated points as update_lists will deal with the
        # new sizes of stories.
        return self.sprints.at[self.current_sprint, "points"]

    def calculate_regressors(self):
        """Calculate the regressors for the previous sprints."""
        stories = self.original_stories[self.original_stories.sprint_id < self.current_sprint]
        sprints = self.original_sprints[self.original_sprints.index < self.current_sprint]
        stories_regressor = linregress(stories.estimated_size, stories.real_size)
        sprints_regressor = linregress(sprints.points, sprints.real_length)

        # Record the tape
        self.regressor_tape.loc[self.current_sprint, :] = [
            stories_regressor.slope, stories_regressor.intercept, sprints_regressor.slope,
            sprints_regressor.intercept
        ]

        return stories_regressor, sprints_regressor

    def update_sprint_values(self):
        """
        Shift current sprint points and its stories' estimated sizes by some amount determined
        by the regressors calculated on historical data.

        Pretty much to what happened with ShiftedStories, there's an important turnover on story
        points and sprints, -314k -> 112k and 2k -> -2k, where the negatives represent
        overshooting.
        """
        stories_regressor, sprints_regressor = self.calculate_regressors()

        current_sprint_stories = self.stories[self.stories.sprint_id == self.current_sprint]

        # rounding up to the next available integer improves the accuracy
        new_values = (
            current_sprint_stories.estimated_size
            * stories_regressor.slope
            + stories_regressor.intercept
        )
        # new_values
        self.stories.loc[current_sprint_stories.index, "estimated_size"] = new_values.astype(int) + 1

        # and the sprint estimation
        new_points = (
            self.sprints.at[self.current_sprint, "points"]
            * sprints_regressor.slope
            + sprints_regressor.intercept
        )
        self.sprints.at[self.current_sprint, "points"] = new_points
