from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple, Union, cast
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import BaseCrossValidator

from mapie.conformity_scores import BaseRegressionScore
from mapie.regression.regression import _MapieRegressor
from mapie.utils import (
    _check_alpha,
    _check_alpha_and_n_samples,
    _transform_confidence_level_to_alpha_list,
    check_is_fitted,
)


class TimeSeriesRegressor(_MapieRegressor):
    """
    Prediction intervals with out-of-fold residuals for time series.
    This class has three valid `method` : `"enbpi"`, `"aci"` or `"spci"`

    The prediction intervals are calibrated on a split of the trained data.
    All strategies are estimating prediction intervals
    on single-output time series.

    EnbPI allows you to update conformal scores using the `update`
    function. It will replace the oldest one with the newest scores.
    It will keep the same amount of total scores

    Actually, EnbPI only corresponds to `TimeSeriesRegressor` if the
    `cv` argument is of type `BlockBootstrap`.

    The ACI strategy allows you to adapt the conformal inference
    (i.e the quantile). If the real values are not in the coverage,
    the size of the intervals will grow. Conversely, if the real values are in
    the coverage, the size of the intervals will decrease. You can use a gamma
    coefficient to adjust the strength of the correction. If the quantile is
    equal to zero, the method will produce an infinite set size.

    The SPCI strategy (Sequential Predictive Conformal Inference) fits a
    quantile random forest on windows of the most recent conformity scores
    (residuals) in order to predict the conditional quantiles of the next
    residual. Since the serial dependence of the residuals is explicitly
    learned, the prediction intervals are typically tighter than those of
    EnbPI and ACI at a similar coverage level. The window length
    (`spci_window`) and the refitting frequency of the residual forest
    (`spci_refit_every`, in number of calls to `update`) can be adjusted;
    both parameters are only used when `method="spci"`. The residual forest
    is a scikit-learn `RandomForestRegressor` whose leaf aggregation
    provides the conditional quantiles, hence no additional dependency
    is required. The width of the prediction intervals is minimized over
    the offset of the two residual quantiles, so the `optimize_beta`
    argument of `predict` has no effect for this method.

    References
    ----------
    Chen Xu, and Yao Xie.
    "Conformal prediction for dynamic time-series."
    https://arxiv.org/abs/2010.09107

    Isaac Gibbs, Emmanuel Candes
    "Adaptive conformal inference under distribution shift"
    https://proceedings.neurips.cc/paper/2021/file/\
0d441de75945e5acbc865406fc9a2559-Paper.pdf

    Margaux Zaffran et al.
    "Adaptive Conformal Predictions for Time Series"
    https://arxiv.org/pdf/2202.07282.pdf

    Chen Xu, and Yao Xie.
    "Sequential Predictive Conformal Inference for Time Series."
    International Conference on Machine Learning (ICML 2023).
    https://arxiv.org/abs/2212.03463
    (clean-room implementation, following issue #370)

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import TimeSeriesRegressor
    >>> X_toy = np.arange(40).reshape(-1, 1)
    >>> y_toy = (5 + 2 * X_toy[:, 0] + np.sin(X_toy[:, 0] / 2)
    ...          + np.random.default_rng(1).normal(scale=0.1, size=40))
    >>> mapie_ts = TimeSeriesRegressor(method="spci", cv=-1, random_state=0)
    >>> mapie_ts = mapie_ts.fit(X_toy, y_toy)
    >>> y_pred, y_pis = mapie_ts.predict(X_toy, confidence_level=0.9)
    >>> y_pis.shape
    (40, 2, 1)
    >>> mapie_ts = mapie_ts.update(X_toy, y_toy)
    >>> y_pred, y_pis = mapie_ts.predict(X_toy, confidence_level=0.9)
    >>> y_pis.shape
    (40, 2, 1)
    """

    cv_need_agg_function_ = _MapieRegressor.cv_need_agg_function_ + ["BlockBootstrap"]
    valid_methods_ = ["enbpi", "aci", "spci"]
    default_sym_ = False
    # Number of trees of the SPCI residual quantile forest, and number of
    # offsets scanned by the SPCI width optimization.
    spci_n_estimators_ = 100
    spci_n_betas_ = 21
    # Minimum number of residuals per leaf of the SPCI residual forest:
    # smoother conditional distributions prevent the width optimization
    # from overfitting noisy leaf estimates.
    spci_min_samples_leaf_ = 10

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "enbpi",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        conformity_score: Optional[BaseRegressionScore] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        spci_window: int = 5,
        spci_refit_every: int = 1,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            agg_function=agg_function,
            verbose=verbose,
            conformity_score=conformity_score,
            random_state=random_state,
        )
        self.spci_window = spci_window
        self.spci_refit_every = spci_refit_every

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> TimeSeriesRegressor:
        """
        Fit estimator, compute conformity scores used for prediction
        intervals and, if `method="spci"`, fit the residual quantile forest.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
            By default `None`.

        kwargs : dict
            Additional fit and predict parameters.

        Returns
        -------
        TimeSeriesRegressor
            The model itself.
        """
        self._check_spci_parameters()
        super().fit(X, y, groups=groups, **kwargs)
        if self.method == "spci":
            self._fit_spci_residual_forest()
            self.n_updates_ = 0
        return self

    def _check_spci_parameters(self) -> None:
        """
        Check the SPCI parameters, and that they are only
        used if `method="spci"`.

        Raises
        ------
        ValueError
            If `spci_window` or `spci_refit_every` is not
            a strictly positive integer, or if one of them is
            set to a non-default value while `method != "spci"`.
        """
        if self.method == "spci":
            for name, value in [
                ("spci_window", self.spci_window),
                ("spci_refit_every", self.spci_refit_every),
            ]:
                if not isinstance(value, (int, np.integer)) or value < 1:
                    raise ValueError(
                        f"Invalid {name}. Allowed values are integers >= 1."
                    )
        elif (self.spci_window != 5) or (self.spci_refit_every != 1):
            raise ValueError(
                "spci_window and spci_refit_every "
                "can be used only with method='spci'."
            )

    def _fit_spci_residual_forest(self) -> None:
        """
        Fit the random forest used by the `"spci"` method on windows of past
        conformity scores (residuals): each training sample of the forest
        contains the `spci_window_` most recent residuals before a time step,
        and targets the residual at this time step. Non-finite residuals
        (for instance not calibrated in a `"split"` cross-validation)
        are discarded.
        """
        residuals = np.asarray(self.conformity_scores_, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) < 2:
            raise ValueError(
                "SPCI requires at least two finite conformity scores "
                "to fit the residual quantile forest."
            )
        # The window is capped so that at least half of the residuals
        # remain as training targets of the forest.
        window = int(min(self.spci_window, len(residuals) // 2))
        X_windows = np.array(
            [
                residuals[i : i + window]
                for i in range(len(residuals) - window)
            ]
        )
        y_windows = residuals[window:]
        self.spci_window_ = window
        self.spci_residuals_ = y_windows
        self.spci_forest_ = RandomForestRegressor(
            n_estimators=self.spci_n_estimators_,
            min_samples_leaf=self.spci_min_samples_leaf_,
            random_state=self.random_state,
        ).fit(X_windows, y_windows)
        self.spci_leaves_ = self.spci_forest_.apply(X_windows)

    def _spci_residual_quantiles(self, levels: NDArray) -> NDArray:
        """
        Estimate the quantiles of the next residual, conditional on the most
        recent residuals, with the leaf-aggregation quantile random forest:
        training residuals that share a leaf with the query window are pooled
        across trees and the weighted empirical quantiles are returned.

        Parameters
        ----------
        levels: NDArray of shape (n_levels,)
            The quantile levels to estimate, between 0 and 1.

        Returns
        -------
        NDArray of shape (n_levels,)
            The estimated conditional quantiles of the next residual.
        """
        query = np.asarray(self.conformity_scores_[-self.spci_window_ :])
        query_leaves = self.spci_forest_.apply(query.reshape(1, -1))[0]
        weights = np.zeros_like(self.spci_residuals_, dtype=float)
        for tree_ix, leaf in enumerate(query_leaves):
            in_leaf = self.spci_leaves_[:, tree_ix] == leaf
            weights[in_leaf] += 1.0 / in_leaf.sum()
        weights /= len(query_leaves)
        order = np.argsort(self.spci_residuals_)
        sorted_residuals = self.spci_residuals_[order]
        cum_weights = np.cumsum(weights[order])
        cum_weights /= cum_weights[-1]
        indices = np.searchsorted(cum_weights, levels, side="left")
        return sorted_residuals[indices]

    def _spci_residual_interval(
        self, alpha: NDArray, allow_infinite_bounds: bool
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the width-optimized residual quantiles used by the `"spci"`
        method: for each risk level `alpha`, the offset `beta` is scanned in
        `[0, alpha]` and chosen such that the distance between the residual
        quantiles at levels `beta` and `1 - alpha + beta` is minimal.

        Parameters
        ----------
        alpha: NDArray of shape (n_alpha,)
            Between `0` and `1`, represents the uncertainty of the
            confidence interval.

        allow_infinite_bounds: bool
            Whether to skip the check on the number of residual
            forest training samples.

        Returns
        -------
        Tuple[NDArray, NDArray]
            The lower and upper residual quantiles,
            both of shape (n_alpha,).
        """
        if not allow_infinite_bounds:
            _check_alpha_and_n_samples(alpha, len(self.spci_residuals_))
        lower_quantiles = np.empty(len(alpha))
        upper_quantiles = np.empty(len(alpha))
        betas = np.linspace(0.0, 1.0, self.spci_n_betas_)
        for alpha_ix, alpha_ in enumerate(alpha):
            levels = np.concatenate([betas * alpha_, 1.0 - alpha_ + betas * alpha_])
            quantiles = self._spci_residual_quantiles(levels)
            widths = quantiles[len(betas) :] - quantiles[: len(betas)]
            best_ix = int(np.argmin(widths))
            lower_quantiles[alpha_ix] = quantiles[best_ix]
            upper_quantiles[alpha_ix] = quantiles[len(betas) + best_ix]
        return lower_quantiles, upper_quantiles

    def _relative_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensemble: bool = False,
    ) -> NDArray:
        """
        Compute the conformity scores on a data set.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
                Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If `False`, predictions are those of the model trained on the
            whole training set.
            If `True`, predictions from perturbed models are aggregated by
            the aggregation function specified in the `agg_function`
            attribute.
            If `cv` is `"prefit"` or `"split"`, `ensemble` is ignored.

            By default `False`.

        Returns
        -------
            The conformity scores corresponding to the input data set.
        """
        y_pred = super().predict(X, ensemble=ensemble)
        scores = np.array(
            self.conformity_score_function_.get_conformity_scores(y, y_pred, X=X)
        )
        return scores

    def _update_conformity_scores_with_ensemble(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensemble: bool = False,
    ) -> TimeSeriesRegressor:
        """
        Update the `conformity_scores_` attribute when new data with known
        labels are available.
        Note: Don't use `_update_conformity_scores_with_ensemble` with samples of the training set.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If `False`, predictions are those of the model trained on the
            whole training set.
            If `True`, predictions from perturbed models are aggregated by
            the aggregation function specified in the `agg_function`
            attribute.
            If `cv` is `"prefit"` or `"split"`, `ensemble` is ignored.

            By default `False`.

        Returns
        -------
        TimeSeriesRegressor
            The model itself.

        Raises
        ------
        ValueError
            If the length of `y` is greater than
            the length of the training set.
        """
        check_is_fitted(self)
        X, y = cast(NDArray, X), cast(NDArray, y)
        m, n = len(X), len(self.conformity_scores_)
        if m > n:
            raise ValueError(
                "The number of observations to update is higher than the"
                "number of training instances."
            )
        new_conformity_scores_ = self._relative_conformity_scores(
            X, y, ensemble=ensemble
        )
        self.conformity_scores_ = np.roll(
            self.conformity_scores_, -len(new_conformity_scores_)
        )
        self.conformity_scores_[-len(new_conformity_scores_) :] = new_conformity_scores_
        return self

    @staticmethod
    def _check_gamma(gamma: float) -> None:
        """
        Check if gamma is between 0 and 1.

        Parameters
        ----------
        gamma: float

        Raises
        ------
        ValueError
            If gamma is lower than 0 or higher than 1.
        """
        if (gamma < 0) or (gamma > 1):
            raise ValueError("Invalid gamma. Allowed values are between 0 and 1.")

    def _get_alpha(
        self, alpha: Optional[Union[float, Iterable[float]]] = None, reset: bool = False
    ) -> Optional[NDArray]:
        """
        Get and set the current alpha (or confidence_level) value(s) given the
        initial alpha (or confidence_level) value(s) for ACI method.

        This method retrieves the alpha value(s) used for confidence intervals.
        If the alpha value(s) is provided, it returns the current alpha
        value(s) stored in the object. Else, nothing. If the reset flag is set
        to True, it resets the current alpha value(s).

        Parameters
        ----------
        alpha: Optional[NDArray]
            Between `0` and `1`, represents the uncertainty of the
            confidence interval.

            By default `None`.

        reset: bool
            Flag indicating whether to reset the current alpha value(s).

        Returns
        -------
        Optional[Union[float, Iterable[float]]]
            The current alpha value(s) for confidence intervals.
        """
        if "current_alpha" not in self.__dict__ or reset:
            self.current_alpha: dict[float, float] = {}

        if alpha is not None:
            alpha_np = cast(NDArray, _check_alpha(alpha))
            alpha_np = np.round(alpha_np, 2)
            for ix, alpha_checked in enumerate(alpha_np):
                alpha_np[ix] = self.current_alpha.setdefault(
                    alpha_checked, alpha_checked
                )
            alpha = alpha_np
        return alpha

    def adapt_conformal_inference(
        self,
        X: ArrayLike,
        y: ArrayLike,
        gamma: float,
        confidence_level: Optional[Union[float, Iterable[float]]] = None,
        ensemble: bool = False,
        optimize_beta: bool = False,
    ) -> TimeSeriesRegressor:
        """
        Adapt the `alpha_t` attribute when new data with known
        labels are available.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If `False`, predictions are those of the model trained on the
            whole training set.
            If `True`, predictions from perturbed models are aggregated by
            the aggregation function specified in the `agg_function`
            attribute.
            If `cv` is `"prefit"` or `"split"`, `ensemble` is ignored.

            By default `False`.

        gamma: float
            Coefficient that decides the correction of the conformal inference.
            If it equals 0, there are no corrections.

        confidence_level: Optional[Union[float, Iterable[float]]]
            Between `0` and `1`, represents the confidence level of the interval.

            By default `None`.

        optimize_beta: bool
            Whether to optimize the PIs' width or not.

            By default `False`.

        Returns
        -------
        TimeSeriesRegressor
            The model itself.

        Raises
        ------
        ValueError
            If the length of `y` is greater than
            the length of the training set.
        """
        if self.method != "aci":
            raise AttributeError(
                "This method can be called only with method='aci', "
                f"not with '{self.method}'."
            )

        check_is_fitted(self)
        self._check_gamma(gamma)
        X, y = cast(NDArray, X), cast(NDArray, y)

        self._get_alpha()
        alpha = self._transform_confidence_level_to_alpha_array(confidence_level)
        if alpha is None:
            alpha = np.array(list(self.current_alpha.keys()))
        alpha_np = cast(NDArray, alpha)

        for x_row, y_row in zip(X, y):
            x = np.expand_dims(x_row, axis=0)
            _, y_pred_bounds = self.predict(
                x,
                ensemble=ensemble,
                confidence_level=1 - alpha_np,
                optimize_beta=optimize_beta,
                allow_infinite_bounds=True,
            )

            for alpha_ix, alpha_0 in enumerate(alpha_np):
                alpha_t = self.current_alpha[alpha_0]

                is_lower_bounded = y_row > y_pred_bounds[:, 0, alpha_ix]
                is_upper_bounded = y_row < y_pred_bounds[:, 1, alpha_ix]
                is_not_bounded = not (is_lower_bounded and is_upper_bounded)

                new_alpha_t = alpha_t + gamma * (alpha_0 - is_not_bounded)
                self.current_alpha[alpha_0] = np.clip(new_alpha_t, 0, 1)

        return self

    def update(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensemble: bool = False,
        confidence_level: Optional[Union[float, Iterable[float]]] = None,
        gamma: float = 0.0,
        optimize_beta: bool = False,
    ) -> TimeSeriesRegressor:
        """
        Update conformity scores

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If `False`, predictions are those of the model trained on the
            whole training set.
            If `True`, predictions from perturbed models are aggregated by
            the aggregation function specified in the `agg_function`
            attribute.
            If `cv` is `"prefit"` or `"split"`, `ensemble` is ignored.

            By default `False`.

        confidence_level: Optional[Union[float, Iterable[float]]]
            (deprecated)
            Between `0` and `1`, represents the confidence level of the interval.

            By default `None`.

        gamma: float
            (deprecated)
            Coefficient that decides the correction of the conformal inference.
            If it equals 0, there are no corrections.

            By default `0.`.

        optimize_beta: bool
            (deprecated)
            Whether to optimize the PIs' width or not.

            By default `False`.

        Returns
        -------
        TimeSeriesRegressor
            The model itself.

        Raises
        ------
        ValueError
            If the length of `y` is greater than
            the length of the training set.
        """
        warn("""
        This function behavior has been changed to allow updating the scores even when using ACI.
        Currently the parameters confidence_level and optimize_beta have no effect. They are kept
        for API stability and will be removed in a future release.
        If you want to adapt confidence level, use adapt_conformal_inference instead.
        """)
        self._check_method(self.method)
        if self.method not in ["enbpi", "aci", "spci"]:
            raise ValueError(
                f"Invalid method. Allowed values are {self.valid_methods_}."
            )

        self._update_conformity_scores_with_ensemble(X, y, ensemble=ensemble)
        if self.method == "spci":
            self.n_updates_ += 1
            if self.n_updates_ % self.spci_refit_every == 0:
                self._fit_spci_residual_forest()
        return self

    # Overriding _MapieRegressor .predict method here. Bad practise, but this
    # inheritance is questionable and will probably be reconsidered anyway.
    def predict(  # type: ignore[override]
        self,
        X: ArrayLike,
        ensemble: bool = False,
        confidence_level: Optional[Union[float, Iterable[float]]] = None,
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        **predict_params,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If `False`, predictions are those of the model trained on the
            whole training set.
            If `True`, predictions from perturbed models are aggregated by
            the aggregation function specified in the `agg_function`
            attribute.
            If `cv` is `"prefit"` or `"split"`, `ensemble` is ignored.

            By default `False`.

        confidence_level: Optional[Union[float, Iterable[float]]]
            Between `0` and `1`, represents the confidence level of the interval.

            By default `None`.

        optimize_beta: bool
            Whether to optimize the PIs' width or not.
            Has no effect if `method="spci"`, whose interval width
            is always optimized.

            By default `False`.

        allow_infinite_bounds: bool
            Allow infinite prediction intervals to be produced.

        predict_params : dict
            Additional predict parameters.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - NDArray of shape (n_samples,) if `alpha` is `None`.
            - Tuple[NDArray, NDArray] of shapes (n_samples,) and
              (n_samples, 2, n_alpha) if `alpha` is not `None`.
              - [:, 0, :]: Lower bound of the prediction interval.
              - [:, 1, :]: Upper bound of the prediction interval.
        """
        alpha = self._transform_confidence_level_to_alpha_array(confidence_level)
        if alpha is None:
            super().predict(
                X,
                ensemble=ensemble,
                alpha=alpha,
                optimize_beta=optimize_beta,
                **predict_params,
            )
        if self.method == "aci":
            alpha = self._get_alpha(alpha)

        if self.method == "spci" and alpha is not None:
            y_pred = np.asarray(
                super().predict(X, ensemble=ensemble, alpha=None, **predict_params)
            )
            residual_low, residual_up = self._spci_residual_interval(
                alpha, allow_infinite_bounds
            )
            y_pred_low = y_pred[:, np.newaxis] + residual_low[np.newaxis, :]
            y_pred_up = y_pred[:, np.newaxis] + residual_up[np.newaxis, :]
            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)

        return super().predict(
            X,
            ensemble=ensemble,
            alpha=alpha,
            optimize_beta=optimize_beta,
            allow_infinite_bounds=allow_infinite_bounds,
            **predict_params,
        )

    # The public API changed from alpha to confidence_level.
    # TODO: refactor this class to use confidence_level everywhere
    @staticmethod
    def _transform_confidence_level_to_alpha_array(
        confidence_level: Optional[Union[float, Iterable[float]]] = None,
    ) -> Optional[NDArray]:
        confidence_level = cast(Optional[NDArray], _check_alpha(confidence_level))
        if confidence_level is None:
            alpha = None
        else:
            alpha = np.array(
                _transform_confidence_level_to_alpha_list(confidence_level)
            )
        return alpha

    @property
    def conformity_scores(self) -> NDArray:
        """
        Returns the conformity scores computed by the `fit` method on the
        out-of-resample predictions of the bootstrap ensemble.

        Returns
        -------
        NDArray
            Array of conformity scores, with shape `(n_samples,)`.
        """
        check_is_fitted(self)
        return cast(NDArray, self.conformity_scores_)
