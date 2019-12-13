import numpy as np


class CorrespondenceMatrixReconstructor:
    """Use the Sinkhorn method to reconstruct a correspondence matrix.

    Attributes
    ----------
    cost_function : callable
        A scalar function to compute cost from time and distance
        (it also uses the parameters alpha and beta, see below).
    alpha : float
        Parameter to be used for cost computation.
    beta : float
        Parameter to be used for cost computation.
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

    def __init__(self, cost_function, alpha, beta, C,
                 living_people, working_people,
                 max_iters=1000, stopping_eps=0.001):
        if len(living_people.shape) != 1 or len(working_people.shape) != 1:
            raise ValueError('living_people (L) and working_people (W) '
                             'should be 1-D arrays')
        if living_people.sum() != working_people.sum():
            raise ValueError('The number of all people (N) is contradictory')
        if max_iters <= 0.0:
            raise ValueError('max_iters should be a positive integer')
        if stopping_eps <= 0.0:
            raise ValueError('stopping_eps should be positive')

        self.cost_function = np.vectorize(cost_function)
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.living_people = living_people
        self.working_people = working_people
        self.max_iters = max_iters
        self.stopping_eps = stopping_eps
        self.cost_matrix = None

    def fit(self, cost_matrix_time, cost_matrix_distance):
        """Compute one cost matrix from 2 given matrices.

        Parameters
        ----------
        cost_matrix_time : array of shape (n_areas, n_areas)
            Time to go from one area to another.
        cost_matrix_distance : array of shape (n_areas, n_areas)
            Distances between areas.

        Returns
        -------
        self : returns an instance of self.
        """
        if (len(cost_matrix_time.shape) != 2 or
                len(cost_matrix_distance.shape) != 2):
            raise ValueError('cost_matrix_time and cost_matrix_distance '
                             'should be 2-D arrays')
        if cost_matrix_time.shape != cost_matrix_distance.shape:
            raise ValueError('Shapes of cost_matrix_time '
                             'and cost_matrix_distance do not match')

        self.cost_matrix = self.cost_function(
            alpha=self.alpha, beta=self.beta,
            time=cost_matrix_distance, distance=cost_matrix_distance
        )
        self.cost_matrix = np.nan_to_num(self.cost_matrix, nan=np.inf)
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
            print('\niter_idx =', iter_idx)
            if iter_idx % 2 == 0:
                new_lambdas_w = lambdas_w
                new_lambdas_l = np.log(np.sum(
                    (np.exp(-lambdas_w - 1 - self.cost_matrix / self.C)).T
                    / norm_living_people, axis=0
                ))
            else:
                new_lambdas_l = lambdas_l
                new_lambdas_w = np.log(np.sum(
                    (np.exp(-lambdas_l - 1 - self.cost_matrix / self.C)).T
                    / norm_working_people, axis=0
                ))

            deltas = np.concatenate((
                new_lambdas_l - lambdas_l, new_lambdas_w - lambdas_w
            ))
            if np.linalg.norm(deltas) < self.stopping_eps:
                break
            lambdas_l = new_lambdas_l
            lambdas_w = new_lambdas_w
            print('lambdas_l =', lambdas_l)
            print('lambdas_w =', lambdas_w)

        reconstructed_correspondence_matrix = (
            np.exp(-1 - self.cost_matrix / self.C) *
            np.exp(-lambdas_l - np.reshape(lambdas_w, (n_areas, 1)))
        )
        reconstructed_correspondence_matrix *= n_people
        return reconstructed_correspondence_matrix

