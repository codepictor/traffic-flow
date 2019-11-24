import os
import os.path

import traffic_data
import utils


def main():
    data = traffic_data.TrafficData()
    data.parse(os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        '..', 'data', 'data.csv'
    ))
    utils.plot_matrix(data.correspondence_matrix, 'correspondence_matrix')
    utils.plot_matrix(data.cost_matrix_time, 'cost_matrix_time')
    utils.plot_matrix(data.cost_matrix_distance, 'cost_matrix_distance')


if __name__ == '__main__':
    main()
