from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize


class Parameter(ABC):
    """
    A static class that supports parameter fitting.

    Attributes:
        name: The name of this parameter.
        bounds: The bounds of this parameter assuming normalized data.

    methods:
        init(x, y): Function that computes the initial value of this
              parameter.
            x: The data positions.
            y: The data weights.
        prior(par, par_0): The prior penalty function for this parameter.
            par: The current value of the parameter.
            par_0: The initial value of the parameter.
        denormalize(par, x, y): Denormalize the parameter where x and y
              have been scaled to fit between 0 and 1.
    """

    name = ""
    bounds = (None, None)

    @staticmethod
    @abstractmethod
    def init(x, y): ...

    @staticmethod
    @abstractmethod
    def prior(par, par_0): ...

    @staticmethod
    @abstractmethod
    def scale(par, x, y): ...


def compute_parameter_covariance(curve, param_names, pos, data, fitp):
    """Compute the parameter covariance matrix at the fit optimum.

    Uses the same approach as the MATLAB ``util_processFit``:
    ``pcov = pinv(J'J) * (chi²/NDF)`` where *J* is the Jacobian of the
    model curve evaluated at the best-fit parameters and *NDF* is
    ``max(N - p, 1)``.

    Returns an ``(n_params, n_params)`` covariance matrix.
    """
    curve_params = {k: v for k, v in fitp.items() if k in param_names}
    params_vec = np.array([curve_params[n] for n in param_names])
    n_data = len(pos)
    n_params = len(params_vec)

    def _model(p):
        return curve(pos, **dict(zip(param_names, p)))

    # Central finite-difference Jacobian
    f0 = _model(params_vec)
    J = np.empty((n_data, n_params))
    for j in range(n_params):
        h = max(abs(params_vec[j]) * 1e-8, 1e-10)
        p_plus = params_vec.copy()
        p_minus = params_vec.copy()
        p_plus[j] += h
        p_minus[j] -= h
        J[:, j] = (_model(p_plus) - _model(p_minus)) / (2 * h)

    residuals = data - f0
    mse = np.sum(residuals**2) / max(n_data - n_params, 1)
    JtJ = J.T @ J

    try:
        pcov = np.linalg.pinv(JtJ) * mse
    except np.linalg.LinAlgError:
        pcov = np.full((n_params, n_params), np.inf)

    return pcov


def param_fit(curve, params, pos, data, use_prior=False):
    """
    Given a curve function and parameter objects, computes a 1D curve fit
    using Maximum A Postiori (MAP) fitting.

    If `use_prior` is True, prior distributions defined by each parameter will be
    used to penalize parameter values that have a low prior likelihood.
    If `use_prior` is False, optimization is not weighted by
    prior parameter distributions, reducing fitting to Maximum Likelihood
    Estimation (MLE), commonly referred to as least-squares fitting.

    Arguments:
        curve (Callable[x, params]): The curve to be fit.
            x: The data positions.
            params: A list of parameters scipy.optimipze.minimize will fit.
        params (list[Parameter]): A list of Parameter objects. Indicies
              correspond to the params arguments in curve and prior.
        pos (np.array[float]): The data positions.
        data (np.array[float]): The data weights.
        use_prior (bool): Flag to apply the prior penalty in MAP fit.
    """
    x = (pos - np.min(pos)) / (np.max(pos) - np.min(pos))
    y = (data - np.min(data)) / (np.max(data) - np.min(data))
    init = [p.init(x, y) for p in params]

    def forward(x, vec):
        return curve(x, *vec)

    if use_prior:

        def prior(vec):
            return np.sum([p.prior(v) for v, p in zip(vec, params)])
    else:
        prior = None

    bounds = tuple(p.bounds for p in params)
    res = map_fit(forward, x, y, init, bounds, prior)

    fitp = {p.name: p.scale(val, pos, data) for p, val in zip(params, res.x)}
    error = np.sqrt(np.mean((data - curve(pos, **fitp)) ** 2))
    fitp["error"] = error

    param_names = [p.name for p in params]
    pcov = compute_parameter_covariance(curve, param_names, pos, data, fitp)
    parstd = np.sqrt(np.diag(pcov))
    for name, std in zip(param_names, parstd):
        fitp[f"{name}_err"] = float(std)

    return fitp


def map_fit(curve, x, y, init, bounds=None, prior=None):
    """
    Computes a 1D curve fit using Maximum A Postiori (MAP) fitting. If `prior`
    is None this function assumes a uniform prior distribution for all
    parameters, equivelent to Maximum Likelihood Estimation (least-squares).
    If given, the `prior` callable adds a penalty term to optimization that
    encourages the fit to match the maximum prior likelihood of each parameter.

    Arguments:
        curve (Callable[x, params]): The curve to be fit.
            x: The data positions.
            params: A list of parameters scipy.optimize.minimize will fit.
        x (np.array[float]): The data positions.
        y (np.array[float]): The data weights.
        init (list[float]): The initial parameter estimation. Indicies
              correspond to the params arguments in curve and prior.
        bounds (tuple(tuple[float, float])): Boundaries for the fitted params.
        prior (Callable[params]): The penalty function that biases fitting towards
              parameter values with high prior likelihood. If not provided,
              optimization is reduced to Maximum Likelihood Estimation (MLE).
            params: A list of parameter values scipy.optimize.minimize will fit.

    Out:
        res: scipy OptimizeResult object.
    """
    if prior is None:

        def prior(p):
            return 0

    def loss(params, x, y):
        v = np.sum((y - curve(x, params)) ** 2)
        v -= prior(params)
        return v

    res = scipy.optimize.minimize(
        loss,
        init,
        args=(x, y),
        bounds=bounds,
    )
    return res
