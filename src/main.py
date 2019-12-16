import os
import os.path

import numpy as np

import correspondence_matrix_reconstructor as cmr
import cost_function
import traffic_data
import utils


def get_traffic_data():
    data = traffic_data.TrafficData()
    data.parse(os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        '..', 'data', 'data.csv'
    ))
    return data


def plot_traffic_data(data):
    utils.plot_matrix(
        matrix=data.correspondence_matrix,
        plot_name='correspondence_matrix'
    )
    utils.plot_matrix(
        matrix=data.cost_matrix_time,
        plot_name='cost_matrix_time'
    )
    utils.plot_matrix(
        matrix=data.cost_matrix_distance,
        plot_name='cost_matrix_distance'
    )


def main():
    data = get_traffic_data()
    plot_traffic_data(data)
    print('Data have been found. Running optimization...\n')

    cost_func = cost_function.compute_cost_func1
    alphas = 0.05 * np.arange(1, 2)
    betas = 0.1 * np.arange(1, 2)
    reconstruction_errors = np.zeros((len(alphas), len(betas)))

    for alpha_idx in range(len(alphas)):
        for beta_idx in range(len(betas)):
            reconstructed_correspondence_matrix = (
                cmr.CorrespondenceMatrixReconstructor(
                    cost_function=cost_func,
                    alpha=alphas[alpha_idx],
                    beta=betas[beta_idx],
                    C=1.0,
                    max_iters=500,
                    stopping_eps=10**(-4)
                ).fit(
                    cost_matrix_time=data.cost_matrix_time,
                    cost_matrix_distance=data.cost_matrix_distance,
                    living_people=np.nansum(data.correspondence_matrix, axis=1),
                    working_people=np.nansum(data.correspondence_matrix, axis=0)
                ).predict()
            )
            reconstruction_errors[alpha_idx, beta_idx] = np.linalg.norm(
                reconstructed_correspondence_matrix -
                np.nan_to_num(data.correspondence_matrix, nan=0.0)
            )
            print(
                'alpha =', alphas[alpha_idx], 'beta =', betas[beta_idx],
                'reconstruction error =',
                reconstruction_errors[alpha_idx, beta_idx], '\n'
            )
            utils.plot_matrix(reconstructed_correspondence_matrix, 'reconstructed_correspondence_matrix')

    print('\nreconstruction_errors =\n', reconstruction_errors, '\n')
    utils.plot_matrix(
        matrix=reconstruction_errors,
        plot_name='reconstruction_errors_' + cost_func.__name__,
        xticklabels=np.round(betas, 2), yticklabels=np.round(alphas, 2),
        xlabel='beta', ylabel='alpha'
    )
    print('Done.')


if __name__ == '__main__':
    main()
