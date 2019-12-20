import os
import os.path
import pickle

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
        plot_name='correspondence_matrix',
        annot=True, linewidths=0.5
    )
    utils.plot_matrix(
        matrix=data.cost_matrix_time,
        plot_name='cost_matrix_time',
        annot=True, linewidths=0.5
    )
    utils.plot_matrix(
        matrix=data.cost_matrix_distance,
        plot_name='cost_matrix_distance',
        annot=True, linewidths=0.5
    )


def reconstruct_correspondence_matrix(data, cost_func, alphas, betas):
    best_matrix = np.zeros((len(alphas), len(betas)))
    reconstruction_errors = np.zeros((len(alphas), len(betas)))
    min_reconstruction_error = np.inf
    best_alpha, best_beta = np.nan, np.nan

    for alpha_idx in range(len(alphas)):
        for beta_idx in range(len(betas)):
            reconstructed_correspondence_matrix = (
                cmr.CorrespondenceMatrixReconstructor(
                    cost_function=cost_func,
                    alpha=alphas[alpha_idx],
                    beta=betas[beta_idx],
                    C=1.0,
                    max_iters=10000,
                    stopping_eps=10**(-4)
                ).fit(
                    cost_matrix_time=data.cost_matrix_time,
                    cost_matrix_distance=data.cost_matrix_distance,
                    living_people=np.nansum(data.correspondence_matrix, axis=1),
                    working_people=np.nansum(data.correspondence_matrix, axis=0)
                ).predict()
            )
            reconstruction_error = np.linalg.norm(
                reconstructed_correspondence_matrix -
                np.nan_to_num(data.correspondence_matrix, nan=0.0)
            )
            reconstruction_errors[alpha_idx, beta_idx] = reconstruction_error
            if reconstruction_error < min_reconstruction_error:
                min_reconstruction_error = reconstruction_error
                best_alpha, best_beta = alphas[alpha_idx], betas[beta_idx]
                best_matrix = reconstructed_correspondence_matrix
            print(
                'alpha =', alphas[alpha_idx], 'beta =', betas[beta_idx],
                'reconstruction error =', reconstruction_error
            )

    with open('reconstruction_errors.pickle', 'wb') as f:
        pickle.dump(reconstruction_errors, f)

    print('\nOptimization has finished.\n')
    print('reconstruction_errors =\n', reconstruction_errors, '\n')
    print('min_reconstruction_error =', min_reconstruction_error)
    utils.plot_matrix(
        matrix=reconstruction_errors,
        plot_name='reconstruction_errors_' + cost_func.__name__,
        annot=False, linewidths=0,
        xticklabels=np.round(betas, 2), yticklabels=np.round(alphas, 2),
        xlabel='beta', ylabel='alpha'
    )
    return best_matrix, best_alpha, best_beta


def main():
    data = get_traffic_data()
    plot_traffic_data(data)
    print('Data have been found. Running optimization...\n')

    reconstructed_correspondence_matrix, best_alpha, best_beta = (
        reconstruct_correspondence_matrix(
            data=data,
            cost_func=cost_function.compute_cost_func1,
            alphas=0.001 * np.arange(1, 501, 1),
            betas=np.arange(1, 2, 1)
        )
    )
    utils.plot_matrix(
        matrix=reconstructed_correspondence_matrix,
        plot_name='reconstructed_correspondence_matrix',
        annot=True, linewidths=0.5
    )

    print('best_alpha =', best_alpha, 'best_beta =', best_beta)
    print('\nDone.')


if __name__ == '__main__':
    main()
