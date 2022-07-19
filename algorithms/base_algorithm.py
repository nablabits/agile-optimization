import pandas as pd


class BaseAlgorithm:
    """A common baseline for all the algorithms."""

    def __init__(self, sprints, stories, compute_size=None):
        """
        Class constructor

        Arguments:
            sprints: the dataframe with the sprints returned by ExperimentEnvironment
            stories: the dataframe with the stories returned by ExperimentEnvironment
            compute_size: the number of sprints that will be used by the algorithm. When
            designing an algorithm it's desirable to deal with a small subset of sprints to quickly
            develop it.

        It also initiates the output indices and the extra points per sprint that we will send to
        ExperimentEvaluator
        """
        self.sprints = sprints.copy()
        self.stories = stories.copy()
        self.current_sprint = 0
        if compute_size:
            self.sprints = self.sprints.loc[:compute_size, :]

        # the outcomes' lists
        self.extra_points = list()
        self.bkp_indices = list()

    def update_lists(self, points):
        """Update the lists with the estimated points."""
        sprint = self.current_sprint
        sprint_stories = self.stories[self.stories.sprint_id == sprint]
        base = sprint_stories[sprint_stories.estimated_size.cumsum() <= points]
        if base.empty:
            # Sometimes happen that we have a sprint whose estimated length is smaller than the
            # first of the stories. For instance sprint #748 has an estimated length of 19 points
            # and the first story is 20 points.
            self.bkp_indices.append(sprint_stories.index.min())
            self.extra_points.append(0)
        else:
            self.bkp_indices.append(base.index.max())
            self.extra_points.append(points - base.estimated_size.sum())

    def estimate_length(self):
        """
        Come up with the estimated length of the sprint.

        Here lives the main logic of the algorithm and therefore for different algorithms we will
        override this method.
        """
        return self.sprints.at[self.current_sprint, "points"]

    def run(self):
        """
        Main routine

        Basically all the algorithms follow this pattern: for each sprint they estimate the real
        length and then compute the index of the story that gets closer to that estimation and add
        1-point-like stories to account for the difference.
        """
        for s in self.sprints.index:
            self.current_sprint = s

            points = self.estimate_length()

            self.update_lists(points)

        return pd.DataFrame(
            {
                "story_id": self.bkp_indices,
                "extra_points": self.extra_points,
            }
        )
