import numpy as np


class TrafficData:
    """A wrapper to keep parsed data specifying true traffic flow.

    Attributes
    ----------
    correspondence_matrix : array of shape = [n_areas, n_areas]
        The matrix containing number of people
        who live in area i and work in area j.
    cost_matrix_time : array of shape = [n_areas, n_areas]
        The matrix containing average time to go from area i to area j.
    cost_matrix_distance : array of shape = [n_areas, n_areas]
        The matrix containing average distance between areas i and j.
    """

    def __init__(self):
        self.correspondence_matrix = None
        self.cost_matrix_time = None
        self.cost_matrix_distance = None

    def parse(self, file_path, delimiter=','):
        """Extract data from a file and prepare them for analysis.

        Data should have the special format with the following columns:
        1. Area i (starting from 1);
        2. Area j (starting from 1);
        3. Number of people who live in area i and work in area j;
        4. Average time to go from area i to area j;
        5. Average distance between areas i and j.

        Parameters
        ----------
        file_path : str
            Path to a file containing raw data to parse.
        delimiter : str
            The string used to separate values.
        """
        if not (isinstance(file_path, str) and isinstance(delimiter, str)):
            raise TypeError('The arguments should be instances of str.')

        raw_data = np.genfromtxt(file_path, delimiter=delimiter)
        if (len(raw_data.shape) != 2
                or raw_data.shape[1] != 5
                or np.max(raw_data[:, 0]) != np.max(raw_data[:, 1])
                or np.min(raw_data[:, 0]) != 1
                or np.min(raw_data[:, 1]) != 1
                or np.min(raw_data) < 0.0):
            raise ValueError('The data in', file_path, 'are inconsistent. '
                             'Check the format of the data.')

        n_areas = int(np.max(raw_data[:, 0]))
        self.correspondence_matrix = np.full(
            (n_areas, n_areas), np.nan, dtype=np.double
        )
        self.cost_matrix_time = np.full(
            (n_areas, n_areas), np.nan, dtype=np.double
        )
        self.cost_matrix_distance = np.full(
            (n_areas, n_areas), np.nan, dtype=np.double
        )
        for raw_data_line in raw_data:
            i, j = int(raw_data_line[0]) - 1, int(raw_data_line[1]) - 1
            self.correspondence_matrix[i, j] = raw_data_line[2]
            self.cost_matrix_time[i, j] = raw_data_line[3]
            self.cost_matrix_distance[i, j] = raw_data_line[4]

