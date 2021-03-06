import numpy as np


class CorrespondenceMatrixReconstructor:
    """Use the Sinkhorn method to reconstruct a correspondence matrix.

    Attributes
    ----------
    C : float
        Ratio between two terms of the optimization problem.
    living_people : array of shape (n_areas,)
        The first constraint of the optimization problem. One number
        corresponds to one area and is equal to the number of people
        living in this area.
    working_people : array of shape (n_areas,)
        The second constraint of the optimization problem. One number
        corresponds to one area and is equal to the number of people
        working in this area.
    max_iters : int
        Maximum number of iterations to perform during optimization.
    stopping_eps : float
        Tolerance for termination.
    cost_matrix : array of shape (n_areas, n_areas)
        Costs to go between areas.
    """

    def __init__(self, C, max_iters=10000, stopping_eps=0.0001):
        if max_iters <= 0.0:
            raise ValueError('max_iters should be a positive integer')
        if stopping_eps <= 0.0:
            raise ValueError('stopping_eps should be positive')

        self.C = C
        self.max_iters = max_iters
        self.stopping_eps = stopping_eps
        self.cost_matrix = None
        self.living_people = None
        self.working_people = None

    def fit(self, cost_matrix, living_people, working_people):
        """Compute one cost matrix from 2 given matrices.

        Parameters
        ----------
        cost_matrix : array of shape (n_areas, n_areas)
            Costs to go between areas.
        living_people : array of shape (n_areas,)
            The first constraint of the optimization problem. One number
            corresponds to one area and is equal to the number of people
            living in this area.
        working_people : array of shape (n_areas,)
            The second constraint of the optimization problem. One number
            corresponds to one area and is equal to the number of people
            working in this area.

        Returns
        -------
        self : returns an instance of self.
        """
        if len(cost_matrix.shape) != 2:
            raise ValueError('cost_matrix should be a 2-D array')
        if len(living_people.shape) != 1 or len(working_people.shape) != 1:
            raise ValueError('living_people (L) and working_people (W) '
                             'should be 1-D arrays')
        if living_people.sum() != working_people.sum():
            raise ValueError('The number of all people (N) is contradictory')

        self.cost_matrix = cost_matrix
        self.living_people = living_people
        self.working_people = working_people
        return self

    def predict(self):
        """Reconstruct a correspondence matrix.

        Returns
        -------
        out : array of shape (n_areas, n_areas)
            Returns reconstructed correspondence matrix.
        """
        if self.cost_matrix is None:
            raise ValueError('The cost_matrix has not been computed. '
                             'Call the fit method at first.')
        assert len(self.cost_matrix.shape) == 2
        assert self.cost_matrix.shape[0] == self.cost_matrix.shape[1]

        n_people = self.living_people.sum()  # N
        norm_living_people = self.living_people / n_people  # l
        norm_working_people = self.working_people / n_people  # w
        n_areas = self.cost_matrix.shape[0]  # n
        lambdas_l = np.zeros(n_areas)  # lambdas^l
        lambdas_w = np.zeros(n_areas)  # lambdas^w

        for iter_idx in range(self.max_iters):
            if iter_idx % 2 == 0:
                new_lambdas_w = lambdas_w
                new_lambdas_l = np.log(np.sum(
                    (np.exp(-lambdas_w - 1 - self.cost_matrix / self.C)).T
                    / norm_living_people, axis=0
                ))
            else:
                new_lambdas_l = lambdas_l
                new_lambdas_w = np.log(np.sum(
                    (np.exp(-lambdas_l - 1 - self.cost_matrix.T / self.C)).T
                    / norm_working_people, axis=0
                ))

            deltas = np.concatenate((
                new_lambdas_l - lambdas_l, new_lambdas_w - lambdas_w
            ))
            if np.linalg.norm(deltas) < self.stopping_eps:
                break
            lambdas_l = new_lambdas_l
            lambdas_w = new_lambdas_w

        reconstructed_correspondence_matrix = np.exp(
            -1 - self.cost_matrix / self.C
            - (np.reshape(lambdas_l, (n_areas, 1)) + lambdas_w)
        )
        reconstructed_correspondence_matrix *= n_people
        return reconstructed_correspondence_matrix

