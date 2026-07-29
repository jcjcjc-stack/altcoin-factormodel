import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def adjusted_r2(r2, rows, feature_count):
    denominator = rows - feature_count - 1
    if denominator <= 0:
        return np.nan
    return 1 - (1 - r2) * (rows - 1) / denominator


def fit_regression_models(model_data, target, feature_columns, split_fraction=0.8):
    split_idx = int(len(model_data) * split_fraction)
    train_data = model_data.iloc[:split_idx]
    test_data = model_data.iloc[split_idx:]

    X_train = sm.add_constant(train_data[feature_columns], has_constant="add")
    y_train = train_data[target]
    X_test = sm.add_constant(test_data[feature_columns], has_constant="add")
    y_test = test_data[target]

    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)

    train_r2 = model.rsquared
    test_r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pred_corr = np.corrcoef(y_test, y_pred)[0, 1]

    cv_r2s = []
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(train_data):
        cv_X_train = sm.add_constant(train_data.iloc[train_idx][feature_columns], has_constant="add")
        cv_y_train = train_data.iloc[train_idx][target]
        cv_X_val = sm.add_constant(train_data.iloc[val_idx][feature_columns], has_constant="add")
        cv_y_val = train_data.iloc[val_idx][target]
        cv_model = sm.OLS(cv_y_train, cv_X_train).fit()
        cv_r2s.append(r2_score(cv_y_val, cv_model.predict(cv_X_val)))

    ridge_X_train = train_data[feature_columns]
    ridge_X_test = test_data[feature_columns]
    ridge_alphas = [0.01, 0.1, 1, 10, 100, 300, 1000, 3000, 10000]
    ridge_cv_results = []

    for alpha in ridge_alphas:
        alpha_cv_r2s = []
        for train_idx, val_idx in tscv.split(train_data):
            cv_X_train = train_data.iloc[train_idx][feature_columns]
            cv_y_train = train_data.iloc[train_idx][target]
            cv_X_val = train_data.iloc[val_idx][feature_columns]
            cv_y_val = train_data.iloc[val_idx][target]
            cv_ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            cv_ridge_model.fit(cv_X_train, cv_y_train)
            alpha_cv_r2s.append(r2_score(cv_y_val, cv_ridge_model.predict(cv_X_val)))
        ridge_cv_results.append({"alpha": alpha, "cv_r2_mean": np.mean(alpha_cv_r2s)})

    ridge_cv_results = pd.DataFrame(ridge_cv_results)
    best_ridge_alpha = ridge_cv_results.loc[ridge_cv_results["cv_r2_mean"].idxmax(), "alpha"]
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=best_ridge_alpha))
    ridge_model.fit(ridge_X_train, y_train)
    ridge_y_pred = ridge_model.predict(ridge_X_test)

    ridge_train_r2 = ridge_model.score(ridge_X_train, y_train)
    ridge_test_r2 = r2_score(y_test, ridge_y_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_y_pred))
    ridge_pred_corr = np.corrcoef(y_test, ridge_y_pred)[0, 1]

    elastic_net_alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    elastic_net_l1_ratios = [0.1, 0.5, 0.9]
    elastic_net_cv_results = []

    for alpha in elastic_net_alphas:
        for l1_ratio in elastic_net_l1_ratios:
            alpha_cv_r2s = []
            for train_idx, val_idx in tscv.split(train_data):
                cv_X_train = train_data.iloc[train_idx][feature_columns]
                cv_y_train = train_data.iloc[train_idx][target]
                cv_X_val = train_data.iloc[val_idx][feature_columns]
                cv_y_val = train_data.iloc[val_idx][target]
                cv_elastic_net_model = make_pipeline(
                    StandardScaler(),
                    ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000),
                )
                cv_elastic_net_model.fit(cv_X_train, cv_y_train)
                alpha_cv_r2s.append(r2_score(cv_y_val, cv_elastic_net_model.predict(cv_X_val)))
            elastic_net_cv_results.append(
                {"alpha": alpha, "l1_ratio": l1_ratio, "cv_r2_mean": np.mean(alpha_cv_r2s)}
            )

    elastic_net_cv_results = pd.DataFrame(elastic_net_cv_results)
    best_elastic_net_params = elastic_net_cv_results.loc[elastic_net_cv_results["cv_r2_mean"].idxmax()]
    elastic_net_model = make_pipeline(
        StandardScaler(),
        ElasticNet(
            alpha=best_elastic_net_params["alpha"],
            l1_ratio=best_elastic_net_params["l1_ratio"],
            max_iter=100000,
        ),
    )
    elastic_net_model.fit(ridge_X_train, y_train)
    elastic_net_y_pred = elastic_net_model.predict(ridge_X_test)

    elastic_net_train_r2 = elastic_net_model.score(ridge_X_train, y_train)
    elastic_net_test_r2 = r2_score(y_test, elastic_net_y_pred)
    elastic_net_rmse = np.sqrt(mean_squared_error(y_test, elastic_net_y_pred))
    elastic_net_pred_corr = np.corrcoef(y_test, elastic_net_y_pred)[0, 1]
    elastic_net_nonzero_coefficients = np.count_nonzero(elastic_net_model.named_steps["elasticnet"].coef_)

    feature_count = len(feature_columns)
    model_accuracy_comparison = pd.DataFrame(
        [
            {
                "model": "OLS",
                "alpha": np.nan,
                "l1_ratio": np.nan,
                "nonzero_coefficients": feature_count,
                "train_r2": train_r2,
                "adjusted_train_r2": adjusted_r2(train_r2, len(y_train), feature_count),
                "test_r2": test_r2,
                "adjusted_test_r2": adjusted_r2(test_r2, len(y_test), feature_count),
                "rmse": rmse,
                "prediction_correlation": pred_corr,
                "cv_r2_mean": np.mean(cv_r2s),
            },
            {
                "model": "Ridge",
                "alpha": best_ridge_alpha,
                "l1_ratio": np.nan,
                "nonzero_coefficients": np.count_nonzero(ridge_model.named_steps["ridge"].coef_),
                "train_r2": ridge_train_r2,
                "adjusted_train_r2": adjusted_r2(ridge_train_r2, len(y_train), feature_count),
                "test_r2": ridge_test_r2,
                "adjusted_test_r2": adjusted_r2(ridge_test_r2, len(y_test), feature_count),
                "rmse": ridge_rmse,
                "prediction_correlation": ridge_pred_corr,
                "cv_r2_mean": ridge_cv_results["cv_r2_mean"].max(),
            },
            {
                "model": "ElasticNet",
                "alpha": best_elastic_net_params["alpha"],
                "l1_ratio": best_elastic_net_params["l1_ratio"],
                "nonzero_coefficients": elastic_net_nonzero_coefficients,
                "train_r2": elastic_net_train_r2,
                "adjusted_train_r2": adjusted_r2(elastic_net_train_r2, len(y_train), feature_count),
                "test_r2": elastic_net_test_r2,
                "adjusted_test_r2": adjusted_r2(elastic_net_test_r2, len(y_test), feature_count),
                "rmse": elastic_net_rmse,
                "prediction_correlation": elastic_net_pred_corr,
                "cv_r2_mean": elastic_net_cv_results["cv_r2_mean"].max(),
            },
        ]
    )

    elastic_net_coefficients = pd.Series(
        elastic_net_model.named_steps["elasticnet"].coef_,
        index=feature_columns,
        name="coefficient",
    )
    elastic_net_selected_coefficients = (
        elastic_net_coefficients[elastic_net_coefficients.ne(0)]
        .rename("coefficient")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    elastic_net_selected_coefficients["abs_coefficient"] = elastic_net_selected_coefficients["coefficient"].abs()
    elastic_net_selected_coefficients = elastic_net_selected_coefficients.sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return {
        "feature_columns": feature_columns,
        "train_data": train_data,
        "test_data": test_data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "y_pred": y_pred,
        "ridge_model": ridge_model,
        "ridge_y_pred": ridge_y_pred,
        "elastic_net_model": elastic_net_model,
        "elastic_net_y_pred": elastic_net_y_pred,
        "model_accuracy_comparison": model_accuracy_comparison,
        "model_comparison": model_accuracy_comparison.copy(),
        "elastic_net_coefficients": elastic_net_coefficients,
        "elastic_net_selected_coefficients": elastic_net_selected_coefficients,
    }
