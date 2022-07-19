from algorithms import BaseAlgorithm


class E01SimpleAlgorithm(BaseAlgorithm):
    """
    The algorithm that just assumes that every person does 1 point per day and does not care about
    real lengths or sizes.
    """

    def estimate_length(self):
        """Override the main method."""
        return self.sprints.at[self.current_sprint, "points"]
