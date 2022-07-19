from algorithms import BaseAlgorithm


class E02SimpleAlgorithmRecallingMean(BaseAlgorithm):
    """Pretty much like simple algorithm but using the mean of real_length to estimate sprints."""

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
