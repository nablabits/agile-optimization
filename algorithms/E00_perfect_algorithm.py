from algorithms import BaseAlgorithm


class E00PerfectAlgorithm(BaseAlgorithm):
    def estimate_length(self):
        return self.sprints.at[self.current_sprint, "real_length"]

    def update_lists(self, points):
        """
        Update the lists with the estimated points.

        We need to override update lists to make use of stories real_size instead of
        estimated_size.
        """
        sprint = self.current_sprint
        sprint_stories = self.stories[self.stories.sprint_id == sprint]
        base = sprint_stories[sprint_stories.real_size.cumsum() <= points]
        if base.empty:
            # Sometimes happen that we have a sprint whose estimated length is smaller than the
            # first of the stories. For instance sprint #748 has an estimated length of 19 points
            # and the first story is 20 points.
            self.bkp_indices.append(sprint_stories.index.min())
            self.extra_points.append(0)
        else:
            self.bkp_indices.append(base.index.max())
            extra_points = points - base.real_size.sum()
            self.extra_points.append(extra_points)
