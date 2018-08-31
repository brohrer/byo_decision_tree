import numpy as np
import matplotlib.pyplot as plt


class Judge(object):
    """
    This class exists to help create an evaluation function for
    the decision tree. It measures how much deviation from ideal
    behavior exists in a set of days. It is used to determine which
    leaves of the decision tree are most ripe for splitting and
    how that split should occur.

    Because it is a somewhat
    messy calculation and depends on the data history, it needs to
    be initialized with and remember the data set that it is associated
    with. Putting the evaluate() function in its own class is the
    cleanest way to do this.
    """
    def __init__(self, arrival_times_df):
        """
        Parameters
        ----------
        arrival_times_df: DataFrame
        """
        self.arrival_times_df = arrival_times_df

    def find_total_absolute_deviation(self, cols=None):
        """
        For a given set of columns in the Judge's dataframe,
        find the total absolute deviation, in minutes,
        from being perfectly on time.

        Parameters
        ----------
        cols: list of col indices

        Returns
        -------
        total_deviation: int
        """
        if cols is None:
            eval_set_df = self.arrival_times_df
        else:
            eval_set_df = self.arrival_times_df.loc[:, cols]

        departure_time = self.find_departure_time(eval_set_df)
        actual_arrivals = eval_set_df.loc[
            eval_set_df.index == departure_time, :].values
        total_deviation = np.sum(np.abs(actual_arrivals))

        return total_deviation, departure_time

    def find_departure_time(self, eval_set_df):
        """
        Choose a departure time that would only make us late
        one time in ten.

        Parameters
        ----------
        eval_set_df: DataFrame

        Returns
        -------
        departure_time: int
            Minutes before desired arrival to depart, to be on time
            9 times out of 10.
        """
        # Find the 90th percentile lateness for each row
        lateness = eval_set_df.quantile(q=.9, axis=1)

        # Find the departure time that corresponds to a lateness of 0,
        # i.e., the one that gets us there on time 90% of the days.
        lateness[lateness > 0] = -120
        i_dep = np.argmax(lateness.values)
        departure_time = eval_set_df.index[i_dep]
        return departure_time
