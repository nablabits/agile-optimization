import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

from algorithms import BaseAlgorithm
from algorithms.E03_shifted_stories import ShiftedStoryPoints


HS = np.arange(1, 90)


class E05LossFNShifted(BaseAlgorithm):
    """
    This algorithm exploits the fact that the evaluator will use a regret matrix to estimate our
    penalisation and replicates such matrix in the decision. To minimise the regret it accepts a
    `regret_factor` that allows us to make the regret function more or less dramatic.
    """

    def __init__(self, sprints, stories, regret_factor=1.2, compute_size=None):
        super().__init__(sprints, stories, compute_size=compute_size)
        self.regret = self.compute_regret_matrix(regret_factor)
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

    @staticmethod
    def compute_regret_matrix(regret_factor):
        """
        Compute the regret associated to each choice pair size-length.

        Params:
            regret_factor: a float that represents how worse it is to overshoot. Note that for
            regret_factor=1 we equally regret overshooting and undershooting.

        The regret is a value that represents our regret had we chosen the exact length.
        The maximum regret is 0 whereas the minimum, that is negative, happens in the optimal
        decision as expected.
        This is because later we will multiply it by the chances of each sprint length happening
        and, those lengths with 0 probability will be equivalent to max regret. This way we want to
        minimise our regret that is negative.
        """
        x, y = [HS for _ in range(2)]
        vx, vy = np.meshgrid(x, y)

        # get distances where negative values represent overshooting sprints
        regret = vx - vy

        # Compute the regret value for those distances
        condition = regret <= -1
        regret[condition] = np.abs(regret[condition]) ** regret_factor
        return regret - regret.max()  # the max regret will be 0

    def estimate_length(self):
        """
        Get the number of points we will assign to the current sprint.

        We will use the regret matrix to evaluate the regret for each pair of
        (stories_sizes_sum, sprint_length) but not all the pairs have the same probability of
        happening. A way to come up with that probability might be to get the posterior of the
        real_length given a normal distribution of our ability to guess right the length.

        Since gaussian_kde only works with multiple datapoints, in the beginning we have to
        get the prior out of the total set of stories. Then, after a few sprints, we can start
        looking at historic data only
        """
        if self.current_sprint < 10:
            errors = self.sprints.points - self.sprints.real_length
            prior = gaussian_kde(self.sprints.real_length).pdf(HS)
        else:
            history = self.sprints[self.sprints.index <= self.current_sprint]
            errors = history.points - history.real_length
            prior = gaussian_kde(history.real_length).pdf(HS)

        # Calculate the likelihood, it basically says how good are we estimating the sprints'
        # length
        current_length = self.sprints.at[self.current_sprint, "points"]
        likes = norm.pdf(HS, current_length, errors.std())
        posterior = prior * likes
        posterior /= posterior.sum()

        # Multiply each row in the regret matrix with the probability of having that length
        # in the current sprint
        size_probability = (self.regret * posterior).sum(axis=1)
        return size_probability.argmin() - 1
