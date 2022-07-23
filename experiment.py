"""Generate the experiment environment to test optimisation algorithms."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


class ExperimentEnvironment:
    """
    Create the environment that simulates scenarios that happen out there.

    Motivation: We need an environment that provides a common framework for all the algorithms so,
    we can compare their effectiveness.
    There are some attributes that control the shape of the environment:
        - Min and max: the amount of people in the team that can vary depending on the day and the
        hires (that's why they are picked uniformly).
        - Days: a set of days with their probability of happening as most of the sprints we will
        have the full amount of days. In our experiment we are assuming that the sprint lasts two
        weeks in which case the max total days would be 10
    """

    min_people, max_people = 3, 7
    days = pd.Series(range(7, 11), index=[0.03, 0.07, 0.1, 0.8])

    def __init__(self, sprints=5000, no_20=False):
        """
        Class constructor.

        Args:
            sprints: int, define the number of generated sprints. It has no effect if get_env is
            passed use_cache=True.
        """

        self.sprints = sprints
        shortcut_stories = pd.read_csv(
            "source_data/stories_estimations_from_shortcut.csv", index_col=0
        )
        if no_20:
            shortcut_stories = shortcut_stories[shortcut_stories.estimate < 20]
        self.shortcut_stories = shortcut_stories.copy()
        self.no_20 = no_20

    def _draw_days(self):
        """
        Generate samples from a distribution of days given some weights.

        Since 10 days is the most frequent we might want to sample that number most of the time.
        """
        return self.days.sample(
            self.sprints, weights=self.days.index, replace=True
        ).values

    def _get_noisy_sprint(self, days):
        """
        Get a noisy set of sprints for a single person.

        Basically we will use some collected data about the effective time dedicated to burn points
        per sprint (performance). We will assume that the mean of that distribution represents one
        point.

        We need to come up with a shift such that when we face a sprint with ten points and we
        reduce it by a factor of the mean, mu, adding this shift to that reduction will return
        ten again.

        Args:
            days: a numpy array containing the number of days for each sprint.
        Returns: a numpy array with the real length of the sprint for a single person.
        """
        percentages = pd.read_csv("source_data/performance.csv").points
        mu = percentages.mean()
        hs = pd.Series(np.linspace(0, 1, 101))
        weights = gaussian_kde(percentages).pdf(hs.values)
        samples = hs.sample(self.sprints, weights=weights, replace=True).reset_index(
            drop=True
        )
        shift = days * (1 - mu)
        return days * samples.values + shift

    def _get_total_noise_per_sprint(self, days, people):
        """
        Get the total noise per sprint in the set given how many people are in it.

        Each person will draw samples from the distribution created in _get_a_noisy_sprint that
        will represent their effective time in the sprint. Then we will sum the points done by the
        people.
        This has a caveat, as persons individually drawing from the distribution implies
        that each individual is independent of everyone else which feels not true as a week
        with an incident will affect the whole team.

        Args:
            days: a numpy array containing the number of days for each sprint.
            people: a numpy array containing the number of people for each sprint.
        Returns: a numpy array containing the total noise per sprint.
        """
        total_sprint = []
        for _ in range(people):
            individual_sprint = self._get_noisy_sprint(days)
            total_sprint.append(individual_sprint)
        return np.array(total_sprint).sum(axis=0)

    def _get_real_lengths(self, days, people):
        """
        Get the real lengths of the set of sprints.

        Args:
            days: a numpy array containing the number of days for each sprint.
            people: a numpy array containing the number of people for each sprint.
        Returns: a numpy array with the real size for each sprint in integers.
        """
        real_lengths = pd.DataFrame({"people": people, "sprint_len": 0})
        for p in range(people.min(), people.max() + 1):
            people_subset = real_lengths[real_lengths.people == p]
            k1 = self._get_total_noise_per_sprint(days, p)
            real_lengths.loc[people_subset.index, "sprint_len"] = k1[
                people_subset.index
            ]
        assert real_lengths[real_lengths.sprint_len == 0].empty
        return real_lengths.sprint_len.astype(int)

    def plot_sprint_diff(self, use_cache=False):
        """Plot the difference between the estimation and the real length of a sprint."""
        if use_cache:
            data = pd.read_csv(f"cached_experiment/experiment_sprints.csv")
        else:
            data = self._create_sprints()
        diff = data.points - data.real_length
        _, ax = plt.subplots()
        ax.set_title = "Sprint diff estimation vs real"
        ax.set_xlabel = "Sprint no."
        ax.set_ylabel = "Diff in points"
        ax.plot(diff.index, diff.values)

    def _create_sprints(self):
        """Get the main dataframe with the information of each sprint."""
        people = np.random.randint(self.min_people, self.max_people, size=self.sprints)
        days = self._draw_days()
        points = people * days
        return pd.DataFrame(
            {
                "people": people,
                "days": days,
                "points": points,
                "real_length": self._get_real_lengths(days, people),
            },
            index=np.arange(self.sprints),
        )

    def _create_bag_of_stories(self):
        """
        Create a dataframe that will represent the stories available for the sprints.

        Basically, to get the estimated size we will sample from the distribution of story points
        50 stories per sprint pretending that all the stories in the sprint are one-pointers
        exclusively.
        In practice, though, we will likely have more points per story, so this way we ensure
        that the algorithms will have enough room to woefully overshoot. With this setting of
        `self.days.max() + 3` we average ~246 points per sprint worth of stories.

        To avoid errors with indices, a unique id whose values match the index is also returned.

        Returns:
            A pandas DataFrame with a set of stories for each sprint defined by `sprint_id` along
            with the estimated and the real size of them and a unique id.
        """
        max_points_per_sprint = self.max_people * (self.days.max() + 3)
        no_of_samples = max_points_per_sprint * self.sprints
        estimated_size = self.shortcut_stories.sample(
            no_of_samples, replace=True
        ).estimate.values
        sprint_id = np.full(
            (max_points_per_sprint, self.sprints), np.arange(self.sprints)
        ).T.ravel()
        return pd.DataFrame(
            {
                "story_id": np.arange(no_of_samples),
                "sprint_id": sprint_id,
                "estimated_size": estimated_size,
                "real_size": self._get_real_size(estimated_size),
            }
        )

    def _get_noise_for_stories(self):
        """
        Extract the error from the points and the durations.

        The source data contains stories whose duration and estimated
        points are known. Since points are a proxy of time we can compare them as long as they are
        normalised.
        To do so, we scale the durations df to match the max amount of points we ever defined in
        shortcut pretending that the story that last the most should have carried that max amount
        of points.
        Once the durations are normalised in the same units as the score, we can calculate the
        difference that we will use as the noise.

        Returns:
            A numpy array with the differences between the estimation and the real length of the
            story where the negative values represent stories underestimated.
        """
        data = pd.read_csv("source_data/story_points.csv")
        durations = self.shortcut_stories.estimate.max() * data.dur / data.dur.max()
        scores = data.score
        return (durations - scores).round().astype(np.int8)

    def _get_real_size(self, estimated_size):
        """
        Calculate the real size of a story.

        As a starting point we can naively assume that the noise for the stories does not depend on
        the story, this way what we do is for each story in the estimated size array we draw a
        value from the noise bag.
        This will produce that some stories, specially one-pointers, land into negative real
        values, so for these guys we keep drawing samples from the noise bag. This will effectively
        produce that stories are at least 1 point and at most the max points in a story ever had in
        shortcut plus the max noise (something around ~27 points).

        Args:
            A numpy array with the estimated sizes of the stories in the bag of stories.

        Returns:
            A numpy array with the real size corresponding to each story in the bag of stories.
        """
        assert isinstance(estimated_size, np.ndarray)

        # The negative values get exhausted usually after 10 iterations but, we set this number
        # just in case we increase the size of the experiment.
        max_iter = 100

        real_size = np.zeros(estimated_size.size)
        noise_bag = self._get_noise_for_stories()
        under_1 = real_size < 1
        while max_iter and np.sum(under_1):
            entries = estimated_size[under_1]
            noise = noise_bag.sample(entries.shape[0], replace=True).values
            real_size[under_1] = entries + noise
            under_1 = real_size < 1
            max_iter -= 1
        assert np.sum(real_size < 1) == 0, f"Max iterations exhausted, {under_1.shape}"
        return real_size.astype(np.int8)

    def get_env(self, use_cache=False, save=True):
        """
        Get the environment.

        Args:
            use_cache: bool, defines whether a cached experiment should be used, otherwise
            `get_env` will generate a new one.
            save: determine whether a new generated environment should be saved to be used as a
            cache.
        Returns:
            A tuple of dataframes containing the sprints and the stories.
        """
        names = ("sprints", "stories")
        if self.no_20:
            names = ("sprints", "stories_no_20")

        if use_cache:
            sprints, stories = [
                pd.read_csv(f"cached_experiment/experiment_{name}.csv")
                for name in names
            ]
        else:
            sprints = self._create_sprints()
            stories = self._create_bag_of_stories()

            # Ensure that index match the story_ids
            assert (stories.index == stories.story_id).all()

            if save:
                pairs = zip((sprints, stories), names)
                [
                    df.to_csv(f"cached_experiment/experiment_{name}.csv", index=False)
                    for df, name in pairs
                ]
        return sprints, stories


class ExperimentEvaluator:
    """Evaluate the performance of an algorithm."""

    def __init__(self, algorithm_outcome, sprints, stories):
        """
        Class constructor.

        Args:
            algorithm_outcome: a pandas DataFrame containing two columns:
                - story_id: It represents the stories over the experiment's sprints where the
                evaluator should stop at to compute.
                - extra_points: the extra points the evaluator should add on top to fine tune the
                main sum of points.
            sprints: a pandas DataFrame like the one created by ExperimentEnvironment.
            stories: a pandas DataFrame like the one created by ExperimentEnvironment.
        """
        assert algorithm_outcome.shape == (sprints.shape[0], 2)
        assert (algorithm_outcome.extra_points >= 0).all()
        self.outcome = algorithm_outcome.copy()
        self.sprints = sprints
        self.stories = stories

    def _check_extra_points(self):
        """
        Ensure that the extra points are smaller than the next story after the breakpoint.

        This implies that algorithm may not be able to nail the sprint as in the following example:
        sprint.real_length = 10
        algorithm.outcome = [3, 2]  # 3rd index, 2 extra points
        stories up to 3rd index real sum = 8

        However, if stories' 4th index is [1, 5] (estimated_size, real_size) the algorithm will be
        forced to choose [4, 1] which will turn into an overshoot as
        stories up to 4th index real sum = 13; and when we add the extra point we end up
        overshooting by 4.
        """
        next_story_indices = self.outcome.story_id + 1
        sizes = self.stories.loc[next_story_indices, "estimated_size"]
        assert np.all(sizes.values > self.outcome.extra_points.values)

    def _compute_diff(self):
        """
        Compute the value the evaluator will use to compute the regret.

        Remember that sizes refer to stories whereas lengths refer to sprints.

        Most of the columns created in the outcome dataframe will be for debugging purposes as the
        one regret will use will be `distance` but in a nutshell this is the explanation behind the
        logic:
            Basically the distance is the difference between the sprint real length and the sum of
            real sizes of the stories plus the extra points.
            distance = real_length - real_size - extra points
            This effectively means that overshooting returns a negative distance whereas falling
            short a positive one.
            `point_distance` refers to the integer values, that is, distances in terms of story
            points. However, squaring this number turns out in a huge regret so, we might want to
            reduce this distance by some factor that in this case is the mean of the real sizes
            of the environment stories (~3.2).
        """
        estimated_sizes, real_sizes = list(), list()
        for bkp in self.outcome.story_id:
            # locate the sprint_id represented by the breakpoint bkp
            sprint_id = self.stories.at[bkp, "sprint_id"]

            # Locate the first story in that sprint
            start = self.stories[self.stories.sprint_id == sprint_id].index.min()

            # Filter the range
            story_range = self.stories.loc[start:bkp]

            # Compute the estimated_size and real size in the range
            estimated_size = story_range.estimated_size.sum()
            real_size = story_range.real_size.sum()

            # Finally, pack all the stuff
            estimated_sizes.append(estimated_size)
            real_sizes.append(real_size)

        self.outcome["estimated_size"] = estimated_sizes
        self.outcome["real_size"] = real_sizes
        self.outcome["total_real_size"] = (
            self.outcome.real_size + self.outcome.extra_points
        )
        self.outcome["real_length"] = self.sprints.real_length

        self.outcome["point_distance"] = (
            self.outcome.real_length - self.outcome.total_real_size
        )
        self.outcome["distance"] = (
            self.outcome.point_distance / self.stories.real_size.mean()
        )

    def _compute_regret(self):
        """
        Compute the performance value of the algorithm.

        We evaluate the algorithm performance with a number we call regret that represents how much
        we will regret had we hit the sprint's real size with our set of stories + extra points.

        This regret is asymmetrical, as overshooting is worse than falling short. Still, getting
        close to the real size of the sprint, while regretable, is not comparatively as bad as the
        regret for falling short. That's why we square the distance for all distances under 1
        (overshooting the sprint by more than the mean of the real sizes, ~3points)
        https://www.desmos.com/calculator/igkkwcxv0e
        """
        self.outcome["regret"] = self.outcome.distance
        under_1 = self.outcome.distance < 1
        squared_entries = self.outcome[under_1]
        self.outcome.loc[squared_entries.index, "regret"] = (
            squared_entries.distance**2
        )
        self.outcome["regret"] = self.outcome.regret

    def describe(self):
        """Show some stats about the point distances and plot them."""
        self.outcome.point_distance.plot()
        return self.outcome.point_distance.describe()

    def nailed_sprints(self):
        """Return the number of sprints where we land in the optimal capacity."""
        return np.sum(self.outcome.point_distance == 0)

    def sprints_overshoot(self):
        return np.sum(self.outcome.point_distance < 0)

    def accumulated_regret(self):
        """
        Show a plot of how the regret increases over time.

        Ideally we would like this plot to grow logarithmically.
        """
        self.outcome.regret.cumsum().plot()

    def run(self):
        # self._check_extra_points()
        self._compute_diff()
        self._compute_regret()
        return (
            f"Total regret: {self.outcome.regret.sum().astype(int)}; "
            f"Nailed Sprints {self.nailed_sprints()}; "
            f"Sprints overshoot {self.sprints_overshoot()}"
        )


def evaluate_algorithm(algorithm_class, no_20=False, *args, **kwargs):
    """
    The common procedure to evaluate a given algorithm so the process is the same for all of them.
    """
    sprints, stories = ExperimentEnvironment(no_20=no_20).get_env(use_cache=True)  # use the 5k env
    a1 = algorithm_class(sprints, stories, *args, **kwargs)
    algorithm_outcome = a1.run()
    ev = ExperimentEvaluator(algorithm_outcome, sprints, stories)
    outcome = ev.run()
    print(outcome)
    return ev
