import datetime
import requests
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from decision_tree import DecisionTree
from judge import Judge
import tools


def recommend(
    datestrs,
    name='departure_tree',
    verbose=False,
):
    """
    Parameters
    ----------
    datestrs: list of strings
        Datestrings of the format YYYY-MM-DD
    name: string
        The stem of the filename where the model is stored.
    verbose: boolean

    Returns
    -------
    recommendations: dict of int
        Dictionary keys are datestrings and values are departure times.
    """
    model_name = name + '.pickle'
    try:
        # Try to load a saved tree.
        tree = tools.restore(model_name)
    except Exception:
        # If unsuccessful, create a new one.
        tree = create_tree(verbose=verbose)
        tools.store(tree, model_name)

    features_df = create_features(datestrs)
    departures = {}
    for datestr in datestrs:
        estimated_departure = tree.estimate(features_df.loc[datestr, :])
        departures[datestr] = estimated_departure

    return departures


def create_tree(verbose=False):
    """
    Parameters
    ----------
    verbose: boolean

    Returns
    -------
    tree: DecisionTree
    """
    # Load the data.
    trips = get_trips()
    arrival_times_df = get_arrival_times(trips)

    # Assume nan means that the train is late.
    arrival_times_df.fillna(value=30, inplace=True)

    # Split the data into training and testing sets.
    training_dates = []
    tuning_dates = []
    testing_dates = []

    last_training_day = datetime.datetime.strptime('2016-04-30', '%Y-%m-%d')
    last_tuning_day = datetime.datetime.strptime('2017-04-30', '%Y-%m-%d')

    for datestr in arrival_times_df.columns:
        this_date = datetime.datetime.strptime(datestr, '%Y-%m-%d')
        if this_date <= last_training_day:
            training_dates.append(datestr)
        if this_date <= last_tuning_day:
            tuning_dates.append(datestr)
        else:
            testing_dates.append(datestr)

    training_df = arrival_times_df.loc[:, training_dates]
    tuning_df = arrival_times_df.loc[:, tuning_dates]
    testing_df = arrival_times_df.loc[:, testing_dates]

    training_features_df = create_features(list(training_df.columns))
    judge = Judge(training_df)

    # Tune our hyperparameter.
    # Iterate over values for n_min.
    best_tuning_score = 1e10
    best_n_min = 0
    best_tree = None
    for n_min in range(10, 100, 10):

        tree = DecisionTree(
            err_fn=judge.find_total_absolute_deviation, n_min=n_min)
        tree.train(training_features_df)
        training_score = evaluate(tree, training_df)
        tuning_score = evaluate(tree, tuning_df)

        if tuning_score < best_tuning_score:
            best_tuning_score = tuning_score
            best_n_min = n_min
            best_tree = tree

        if verbose:
            print('n_min', n_min)
            print('training', training_score)
            print('tuning', tuning_score)
            tree.render()

    testing_score = evaluate(best_tree, testing_df)
    
    if verbose:
        print('best_n_min', best_n_min)
        print('best_tuning', best_tuning_score)
        print('testing score', testing_score)

    return best_tree


def evaluate(tree, arrivals_df, debug=False):
    """
    Compare the empirical best departure time for each day with
    the model's estimate.

    Parameters
    ----------
    tree: DecisionTree
        The model against which to compare.
    arrivals_df: DataFrame
        The empirical data set against which to compare.
    debug: boolean

    Returns
    -------
    mean_absolute_error: list of floats
        The typical number of minutes between the optimal departure time
        and the one recommended by the model.
    """
    datestrs = list(arrivals_df.columns)
    features = create_features(datestrs)
    deviation = []
    for datestr in datestrs:
        estimated_departure = tree.estimate(features.loc[datestr, :])
        lateness = arrivals_df.loc[:, datestr]
        deviation.append(lateness.loc[estimated_departure])
        if debug:
            print(datestr, 'est', estimated_departure, 'dev', deviation[-1])
    if debug:
        plt.plot(deviation, linestyle='none', marker='.')
        plt.ylabel('minutes late')
        plt.show()

    return np.mean(np.abs(deviation))


def download_data(verbose=True):
    """
    Pull the data down from the public servers.

    Parameters
    ----------
    verbose: boolean

    Returns
    -------
    trips: list of dicts
       Each dictionary has a 'dep' and 'arr' field,
       indicating the departure and arrival datetimes
       for each trip.
    """
    # Harvard Square, Red line stop, outbound
    harvard_stop_id = '70068'
    # JFK / UMass, Red line stop, inbound
    jfk_stop_id = '70086'

    # Gather trip data from a time window from each day,
    # over many days.
    start_time = datetime.time(7, 0)
    end_time = datetime.time(10, 0)
    start_date = datetime.date(2015, 5, 1)
    end_date = datetime.date(2018, 5, 1)

    TTravelURL = "http://realtime.mbta.com/developer/api/v2.1/traveltimes"
    TKey = "?api_key=wX9NwuHnZU2ToO7GmGR9uw"
    TFormat = "&format=json"
    from_stop = "&from_stop=" + str(jfk_stop_id)
    to_stop = "&to_stop=" + str(harvard_stop_id)

    # Cycle through all the days.
    i_day = 0
    trips = []
    while True:
        check_date = start_date + datetime.timedelta(days=i_day)
        if check_date > end_date:
            break

        # Formulate the query.
        from_time = datetime.datetime.combine(check_date, start_time)
        to_time = datetime.datetime.combine(check_date, end_time)
        TFrom_time = "&from_datetime=" + str(int(from_time.timestamp()))
        TTo_time = "&to_datetime=" + str(int(to_time.timestamp()))

        SRequest = "".join([
            TTravelURL,
            TKey,
            TFormat,
            from_stop, to_stop,
            TFrom_time, TTo_time
        ])
        s = requests.get(SRequest)
        s_json = s.json()
        for trip in s_json['travel_times']:
            trips.append({
                'dep': datetime.datetime.fromtimestamp(
                    float(trip['dep_dt'])),
                'arr': datetime.datetime.fromtimestamp(
                    float(trip['arr_dt']))})
        if verbose:
            print(check_date, ':', len(s_json['travel_times']))

        i_day += 1

    return trips


def get_trips():
    """
    Attempt to restore a saved copy.
    If unsuccessful, download a new one.

    Returns
    -------
    trips: list of dictionaries
    """
    trips_filename = 'trips.pickle'
    try:
        trips = tools.restore(trips_filename)
    except Exception:
        trips = download_data()
        tools.store(trips, trips_filename)
    return trips


def get_arrival_times(trips_df):
    """
    Attempt to restore a saved copy.
    If unsuccessful, download a new one.

    Parameters
    ----------
    trips_df: DataFrame

    Returns
    -------
    arrival_times_df: DataFrame
    """
    arrival_times_filename = 'arrival_times.pickle'
    try:
        arrival_times_df = tools.restore(arrival_times_filename)
    except Exception:
        arrival_times_df = None
    if arrival_times_df is None:
        arrival_times_df = calculate_arrival_times(trips_df)
        tools.store(arrival_times_df, arrival_times_filename)

    return arrival_times_df


def calculate_arrival_times(
    trips,
    harvard_walk=4,
    jfk_walk=6,
    target_hour=9,
    target_minute=0,
    train_dep_min=-60,
    train_dep_max=0,
    debug=False,
):
    """
    Based on the downloaded trips data, calculate the arrival times
    that each possible departure time would result in.

    The kwargs above default to our specific use case (work starts
    at 9:00, it takes 6 minutes to walk to JFK, and it takes
    4 minutes to walk from Harvard Square to work)

    Parameters
    ----------
    harvard_walk, jfk_walk: int
        The time in minutes it takes to make these walks.
    trips: DataFrame
    target_hour, target_minute: int
        The time work starts is target_hour:target_minute.
    train_dep_min, train_dep_max: int
        The time, relative to the target, in minutes when the train departs
        from JFK. Negative number means minutes **before** the target.
        Min and max define the time window under consideration.
    debug: boolean
    """
    minutes_per_hour = 60
    date_format = '%Y-%m-%d'
    trips_expanded = []
    for raw_trip in trips:
        rel_dep = (
            minutes_per_hour * (raw_trip['dep'].hour - target_hour) +
            (raw_trip['dep'].minute - target_minute))
        rel_arr = (
            minutes_per_hour * (raw_trip['arr'].hour - target_hour) +
            (raw_trip['arr'].minute - target_minute))

        if rel_dep > train_dep_min and rel_dep <= train_dep_max:
            new_trip = {
                'departure': rel_dep,
                'arrival': rel_arr,
                'date': raw_trip['dep'].date(),
            }
            trips_expanded.append(new_trip)

    trips_df = pd.DataFrame(trips_expanded)

    if debug:
        print(trips_df)
        tools.custom_scatter(trips_df['departure'], trips_df['arrival'])

    door_arrivals = {}
    # Create a new DataFrame with minute-by-minute predictions
    for day in trips_df.loc[:, 'date'].unique():
        datestr = day.strftime(date_format)
        trips_today = trips_df.loc[
            trips_df.loc[:, 'date'] == day, :]
        door_arrival = np.zeros(train_dep_max - train_dep_min)
        for i_row, door_departure in enumerate(
                np.arange(train_dep_min, train_dep_max)):
            # Find the next train departure time.
            station_arrival = door_departure + jfk_walk
            try:

                idx = trips_today.loc[
                    trips_today.loc[:, 'departure'] >=
                    station_arrival, 'departure'].idxmin()
                door_arrival[i_row] = (
                    trips_today.loc[idx, 'arrival'] + harvard_walk)
            except Exception:
                # Fill with not-a-numbers (NaN)
                door_arrival[i_row] = np.nan

        door_arrivals[datestr] = pd.Series(
            door_arrival, index=np.arange(train_dep_min, train_dep_max))
    arrival_times_df = pd.DataFrame(door_arrivals)
    return arrival_times_df

    
def create_features(datestrs):
    """
    Find the features associated with a set of dates.
    These will include:
        weekday / weekend
        day of week
        season
        month of year

    Parameters
    ----------
    datestrs: list of strings
        Date strings of the format YYYY-MM-DD.

    Returns
    -------
    features: DataFrame
        Each row corrsponds to one date. The datestring is the index.
    """
    feature_data = []
    for datestr in datestrs:
        current_date = datetime.datetime.strptime(datestr, '%Y-%m-%d').date()

        current_weekday = current_date.weekday()
        day_of_week = np.zeros(7)
        day_of_week[current_weekday] = 1

        current_month = current_date.month
        month_of_year = np.zeros(12)
        # Adjust months to January = 0
        month_of_year[current_month - 1] = 1

        # Season 0 = winter, 1 = spring, 2 = summer, 3 = autumn
        season = np.zeros(4)
        if current_month <= 2:
            season[0] = 1
        elif current_month <= 5:
            season[1] = 1
        elif current_month <= 8:
            season[2] = 1
        elif current_month <= 11:
            season[3] = 1
        else:
            season[0] = 1

        feature_set = {
            'Saturday': day_of_week[5],
            'Sunday': day_of_week[6],
            'winter': season[0],
            'spring': season[1],
            'summer': season[2],
            'autumn': season[3],
        }
        feature_data.append(feature_set)

    features_df = pd.DataFrame(data=feature_data, index=datestrs)
    return features_df


# Do this when the module is run as a script:
# > python3 departure_alarm.py 2018-08-25 2018-08-26 2018-12-24
if __name__ == '__main__':
    """
    Arguments
    ---------
    strings
        Each command line argument is assumed to be a date string
        of the form YYYY-MM-DD
    """
    recommendations = recommend(sys.argv[1:], verbose=False)
    print(recommendations)
