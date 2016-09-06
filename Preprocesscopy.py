from __future__ import division

import numpy as np
import pandas as pd
from geopy.distance import vincenty
import json
from sklearn.cross_validation import train_test_split


class Preprocesscopy:

    def __init__(self, file_path, city, tile_size, la_size, long_size, freq_threshold):
        self.file_path = file_path
        if city == 'san_francisco':
            self.city_central = (37.773692, -122.420951)
            self.max_la = 37.832373
            self.min_la = 37.708070
            self.max_long = -122.359
            self.min_long = -122.515000
        if city == 'porto':
            self.city_central = (41.162142, -8.621953)

        self.tile_size = tile_size
        self.la_size = la_size
        self.long_size = long_size
        self.freq_threshold = freq_threshold
        self.taxi_trips = []
        self.taxi_train = []
        self.taxi_test = []
        self.region_dict = {}

    def read_dataset(self):
        taxi_trips = pd.read_csv(self.file_path)
        trip_trace = taxi_trips['trip_trace']
        new_triptrace = []
        for i in range(trip_trace.shape[0]):
            new_triptrace.append(json.loads(trip_trace.iloc[i]))

        taxi_trips['trip_trace'] = new_triptrace
        self.taxi_trips = taxi_trips

    def split_train_test(self):
        idxs = np.array(range(self.taxi_trips.shape[0]))
        train_idx, test_idx = train_test_split(idxs, test_size=0.33, random_state=42)

        taxi_train = self.taxi_trips.iloc[train_idx]
        taxi_train = taxi_train.reset_index()
        del taxi_train['index']

        taxi_test = self.taxi_trips.iloc[test_idx]
        taxi_test = taxi_test.reset_index()
        del taxi_test['index']

        self.taxi_train = taxi_train
        self.taxi_test = taxi_test

    # define a function to give the distance between two ( latitude, longtitude) points
    def get_distance(self, start_location, next_location):
        start_ = (start_location[0], start_location[1])
        next_ = (next_location[0], next_location[1])
        return vincenty(start_, next_).miles

    def get_average_speed(self):
        speed_record = []

        for row_index, row in self.taxi_train.iterrows():
            single_trip = row[3]
            trip_length = len(single_trip)

            start_location = single_trip[0][0]
            start_time = single_trip[0][1]

            for i in range(1, trip_length):
                next_location = single_trip[i][0]
                next_time = single_trip[i][1]

                distance = self.get_distance(start_location, next_location)  # in miles
                time_spend = next_time - start_time  # in seconds

                # average speed
                speed_record.append(distance / (time_spend / 3600))  # miles per hour

                start_location = next_location
                start_time = next_time

        speed_record = np.array(speed_record)
        return speed_record.mean(), np.median(speed_record)

    def find_significant_region(self, start_id):

        la_diff = self.max_la - self.min_la
        long_diff = self.max_long - self.min_long

        #number of tiles
        num_la = int(la_diff / self.la_size + 1)
        num_long = int(long_diff / self.long_size + 1)

        occurrence = np.zeros((num_la, num_long))

        for row_index, row in self.taxi_train.iterrows():
            single_trip = row[3]
            trip_length = len(single_trip)

            for i in range(trip_length):
                la_location = single_trip[i][0][0]
                long_location = single_trip[i][0][1]
                x = int((la_location - self.min_la) / self.la_size)
                y = int((long_location - self.min_long) / self.long_size)
                occurrence[x, y] += 1

        # significant tiles is tiles visited  by taxis more than freq_threshold times
        sign_region = {}
        tiles_name = start_id

        for i in range(num_la):
            for j in range(num_long):
                if occurrence[i, j] >= self.freq_threshold:
                    sign_region[(i, j)] = tiles_name
                    tiles_name += 1
                else:
                    continue
        self.region_dict = sign_region

    def filter_low_frequency_trajectory(self):
        new_trace = []
        trace_indicator = []

        for row_index, row in self.taxi_train.iterrows():
            single_trip = row[3]
            trip_length = len(single_trip)
            coordinate_trace = []
            indicator = 0  # 0 means ok, 1 means drop

            for i in range(trip_length):
                la_location = single_trip[i][0][0]
                long_location = single_trip[i][0][1]
                x = int((la_location - self.min_la) / self.la_size)
                y = int((long_location - self.min_long) / self.long_size)
                coordinate = (x, y)
                if coordinate in self.region_dict:
                    coordinate_trace.append(single_trip[i])
                else:
                    indicator = 1
                    break

            trace_indicator.append(indicator)
            new_trace.append(coordinate_trace)

        self.taxi_train['trace_indicator'] = trace_indicator
        self.taxi_train['trip_trace'] = new_trace
        taxi_pandas = self.taxi_train[self.taxi_train['trace_indicator'] == 0]
        del taxi_pandas['trace_indicator']
        taxi_pandas = taxi_pandas.reset_index()
        del taxi_pandas['index']

        self.taxi_train = taxi_pandas

    def refind_significant_region(self, start_id):

        la_diff = self.max_la - self.min_la
        long_diff = self.max_long - self.min_long

        #number of tiles
        num_la = int(la_diff / self.la_size + 1)
        num_long = int(long_diff / self.long_size + 1)

        occurrence = np.zeros((num_la, num_long))

        for row_index, row in self.taxi_train.iterrows():
            single_trip = row[3]
            trip_length = len(single_trip)

            for i in range(trip_length):
                la_location = single_trip[i][0][0]
                long_location = single_trip[i][0][1]
                x = int((la_location - self.min_la) / self.la_size)
                y = int((long_location - self.min_long) / self.long_size)
                occurrence[x, y] += 1

        # significant tiles is tiles visited  by taxis more than freq_threshold times
        sign_region = {}
        tiles_name = start_id

        for i in range(num_la):
            for j in range(num_long):
                if occurrence[i, j] > 0:
                    sign_region[(i, j)] = tiles_name
                    tiles_name += 1
                else:
                    continue
        self.region_dict = sign_region

    def convert_train_traj_to_region_id(self):
        new_trace = []
        trace_indicator = []

        for row_index, row in self.taxi_train.iterrows():
            single_trip = row[3]
            trip_length = len(single_trip)
            tile_trace = []
            indicator = 0  # 0 means ok, 1 means drop

            for i in range(trip_length):
                la_location = single_trip[i][0][0]
                long_location = single_trip[i][0][1]
                x = int((la_location - self.min_la) / self.la_size)
                y = int((long_location - self.min_long) / self.long_size)
                coordinate = (x, y)
                if coordinate in self.region_dict:
                    observe = self.region_dict[coordinate]
                    tile_trace.append(observe)
                else:
                    indicator = 1
                    break

            trace_indicator.append(indicator)
            new_trace.append(tile_trace)

        self.taxi_train['trace_indicator'] = trace_indicator
        self.taxi_train['trip_trace'] = new_trace
        taxi_pandas = self.taxi_train[self.taxi_train['trace_indicator'] == 0]
        del taxi_pandas['trace_indicator']
        taxi_pandas = taxi_pandas.reset_index()
        del taxi_pandas['index']
        self.taxi_train = taxi_pandas

    def convert_test_traj_to_region_id(self):
        new_trace = []
        trace_indicator = []

        for row_index, row in self.taxi_test.iterrows():
            single_trip = row[3]
            trip_length = len(single_trip)
            tile_trace = []
            indicator = 0  # 0 means ok, 1 means drop

            for i in range(trip_length):
                la_location = single_trip[i][0][0]
                long_location = single_trip[i][0][1]
                x = int((la_location - self.min_la) / self.la_size)
                y = int((long_location - self.min_long) / self.long_size)
                coordinate = (x, y)
                if coordinate in self.region_dict:
                    observe = self.region_dict[coordinate]
                    tile_trace.append(observe)
                else:
                    indicator = 1
                    break

            trace_indicator.append(indicator)
            new_trace.append(tile_trace)

        self.taxi_test['trace_indicator'] = trace_indicator
        self.taxi_test['trip_trace'] = new_trace
        taxi_pandas = self.taxi_test[self.taxi_test['trace_indicator'] == 0]
        del taxi_pandas['trace_indicator']
        taxi_pandas = taxi_pandas.reset_index()
        del taxi_pandas['index']
        self.taxi_test = taxi_pandas
