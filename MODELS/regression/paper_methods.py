import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Lasso, lasso_path
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


FEATURES = (
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
)

PAPER_SIX = (
    "Jitter(Abs)", "Shimmer", "NHR", "HNR", "DFA", "PPE",
)

MODELS = ("LS", "IRLS", "LASSO", "CART")

# Explicit implementation grids—not settings supplied by the paper.
LASSO_RATIOS = np.logspace(0, -6, 16)
CART_ALPHAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
LASSO_MAX_ITER = 100_000


class PaperIRLS(RegressorMixin, BaseEstimator):
    """
    No-intercept IRLS using:

        w_i = exp(-2 * abs(residual_i) / max(abs(residual)))

    The initial ordinary least-squares fit counts as iteration one.
    Multiplying rows by sqrt(w) solves:

        sum(w_i * residual_i**2)
    """

    def __init__(self, n_iter=100):
        self.n_iter = n_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)

        if not isinstance(self.n_iter, int) or self.n_iter < 1:
            raise ValueError("n_iter must be a positive integer")

        self.n_features_in_ = X.shape[1]
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        self.n_iter_ = 1

        for iteration in range(1, self.n_iter):
            residual = np.abs(y - X @ self.coef_)
            maximum = residual.max()

            if maximum == 0:
                break

            sqrt_w = np.exp(-residual / maximum)

            self.coef_ = np.linalg.lstsq(
                X * sqrt_w[:, None],
                y * sqrt_w,
                rcond=None,
            )[0]

            self.n_iter_ = iteration + 1

        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Prediction feature count differs from training"
            )

        return X @ self.coef_


def _scale_training(X, scaling):
    if scaling == "none":
        return X, "passthrough"

    if scaling == "maxabs":
        scaler = MaxAbsScaler()
        return scaler.fit_transform(X), scaler

    raise ValueError("scaling must be 'none' or 'maxabs'")


def _transform(scaler, X):
    return X if isinstance(scaler, str) else scaler.transform(X)


def _alpha_max(X, y):
    return max(
        float(np.max(np.abs(X.T @ y)) / len(y)),
        np.finfo(float).eps,
    )


def _lasso_path(X, y):
    return lasso_path(
        X,
        y,
        alphas=_alpha_max(X, y) * LASSO_RATIOS,
        max_iter=LASSO_MAX_ITER,
        tol=1e-6,
    )[1]


def fit_lasso(X, y, inner_folds, scaling):
    """Choose a relative alpha using inner validation only."""

    errors = np.zeros(len(LASSO_RATIOS))

    for train, validation in inner_folds:
        Xt, scaler = _scale_training(X[train], scaling)

        coefficients = _lasso_path(Xt, y[train])

        predictions = (
            _transform(scaler, X[validation]) @ coefficients
        )

        errors += np.abs(
            y[validation, None] - predictions
        ).mean(axis=0)

    best_ratio = LASSO_RATIOS[int(np.argmin(errors))]

    Xt, scaler = _scale_training(X, scaling)

    model = Lasso(
        alpha=_alpha_max(Xt, y) * best_ratio,
        fit_intercept=False,
        max_iter=LASSO_MAX_ITER,
        tol=1e-6,
    ).fit(Xt, y)

    # Both steps are already fitted. Constructing the Pipeline does not refit.
    return Pipeline([
        ("scale", scaler),
        ("regressor", model),
    ])


def fit_pruned_cart(X, y, inner_folds, seed):
    """
    Sklearn cost-complexity pruning.

    This is not a claim of parameter equivalence with MATLAB pruning levels.
    """

    best = None

    for alpha in CART_ALPHAS:
        error = 0.0

        for train, validation in inner_folds:
            tree = DecisionTreeRegressor(
                random_state=seed,
                ccp_alpha=alpha,
                min_samples_leaf=1,
            ).fit(X[train], y[train])

            error += mean_absolute_error(
                y[validation],
                tree.predict(X[validation]),
            )

        # Prefer more pruning when validation errors tie.
        rank = (error / len(inner_folds), -alpha)

        if best is None or rank < best[0]:
            best = (rank, alpha)

    return DecisionTreeRegressor(
        random_state=seed,
        ccp_alpha=best[1],
        min_samples_leaf=1,
    ).fit(X, y)


def information_criteria(
    y,
    prediction,
    n_features,
    reference_variance,
):
    """
    Paper-form RSS/sigma² penalties using ONE shared training variance.

    The paper does not specify a fully reproducible variance-estimation rule.
    Here sigma² is full-feature LS training MSE, held fixed when comparing
    candidate subsets.

    Using each candidate's own RSS/n would cancel the fit-quality term.

    D is the number of input measures, as in the paper—not tree leaf count.
    """

    rss = float(np.sum((y - prediction) ** 2))
    fit_term = rss / reference_variance

    aic = fit_term + 2 * n_features
    bic = fit_term + np.log(len(y)) * n_features

    return aic, bic


def select_feature_sets(X, y, inner_folds, scaling, seed):
    """
    LASSO proposes subsets; pruned CART + training AIC/BIC rank them.

    Every argument contains OUTER-TRAINING data only.

    The published six-feature subset is a separate predefined comparison.
    It is not inserted into the LASSO candidate search.
    """

    Xt, _ = _scale_training(X, scaling)

    full = tuple(range(X.shape[1]))
    candidates = {full}

    for coefficients in _lasso_path(Xt, y).T:
        subset = tuple(
            np.flatnonzero(coefficients != 0.0).tolist()
        )

        if subset:
            candidates.add(subset)

    full_ls = LinearRegression(
        fit_intercept=False
    ).fit(Xt, y)

    residual = y - full_ls.predict(Xt)

    reference_variance = max(
        float(np.mean(residual ** 2)),
        np.finfo(float).eps,
    )

    trees = {}
    criteria = {}

    def assess(subset):
        if subset not in trees:
            matrix = X[:, subset]

            tree = fit_pruned_cart(
                matrix,
                y,
                inner_folds,
                seed,
            )

            trees[subset] = tree

            criteria[subset] = information_criteria(
                y,
                tree.predict(matrix),
                len(subset),
                reference_variance,
            )

        return criteria[subset]

    for subset in sorted(candidates):
        assess(subset)

    aic_subset = min(
        candidates,
        key=lambda s: (criteria[s][0], len(s), s),
    )

    bic_subset = min(
        candidates,
        key=lambda s: (criteria[s][1], len(s), s),
    )

    published = tuple(
        FEATURES.index(name) for name in PAPER_SIX
    )

    assess(published)

    subsets = {
        "all16": full,
        "paper6": published,
        "lasso_aic": aic_subset,
        "lasso_bic": bic_subset,
    }

    return subsets, trees, criteria


def fit_method(name, X, y, inner_folds, scaling):
    if name == "LASSO":
        return fit_lasso(X, y, inner_folds, scaling)

    if name == "LS":
        model = LinearRegression(fit_intercept=False)

    elif name == "IRLS":
        model = PaperIRLS(n_iter=100)

    else:
        raise ValueError(
            "CART is fitted/cached during subset assessment"
        )

    Xt, scaler = _scale_training(X, scaling)
    model.fit(Xt, y)

    return Pipeline([
        ("scale", scaler),
        ("regressor", model),
    ])
                                 