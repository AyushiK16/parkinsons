"""Independent reproduction-style experiment. Does not import Experiment 1."""

from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, ShuffleSplit

from paper_methods import (
    FEATURES,
    MODELS,
    LASSO_RATIOS,
    CART_ALPHAS,
    select_feature_sets,
    fit_method,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "DATA/raw/UCI_Oxford_Voice/telemonitoring/parkinsons_updrs.data"
)

TARGETS = ("motor_UPDRS", "total_UPDRS")
GROUP = "subject#"


def load_data(path):
    df = pd.read_csv(path).reset_index(drop=True)

    required = (
        list(FEATURES)
        + list(TARGETS)
        + [GROUP, "test_time"]
    )

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not np.isfinite(
        df[required].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Required columns must contain finite numeric values"
        )

    return df


def make_inner_folds(X, groups, protocol, count, seed):
    if protocol == "patient":
        count = min(count, len(np.unique(groups)))

        if count < 2:
            raise ValueError(
                "Too few training patients for inner grouped CV"
            )

        folds = list(
            GroupKFold(count).split(X, groups=groups)
        )

        for train, validation in folds:
            assert set(groups[train]).isdisjoint(
                groups[validation]
            )

        return folds

    return list(
        ShuffleSplit(
            n_splits=count,
            test_size=0.2,
            random_state=seed,
        ).split(X)
    )


def run_protocol(
    df,
    protocol,
    repeats,
    group_folds,
    inner_count,
    scaling,
    seed,
):
    X = df[list(FEATURES)].to_numpy(dtype=float)
    groups = df[GROUP].to_numpy()

    if protocol == "recording":
        outer = ShuffleSplit(
            n_splits=repeats,
            test_size=0.1,
            random_state=seed,
        )
        splits = outer.split(X)

    elif protocol == "patient":
        if not 2 <= group_folds <= len(np.unique(groups)):
            raise ValueError("Invalid number of patient folds")

        splits = GroupKFold(group_folds).split(
            X,
            groups=groups,
        )

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    scores = defaultdict(list)
    subsets_seen = defaultdict(Counter)
    selection_scores = defaultdict(list)

    tracking = {
        (target, model): [[] for _ in range(len(df))]
        for target in TARGETS
        for model in ("IRLS", "CART")
    }

    def remember(
        key,
        y_train,
        train_pred,
        y_test,
        test_pred,
        subset,
        criteria,
    ):
        error = y_test - test_pred

        scores[key].append((
            mean_absolute_error(y_train, train_pred),
            mean_absolute_error(y_test, test_pred),
            np.sqrt(mean_squared_error(y_test, test_pred)),
            r2_score(y_test, test_pred),
            len(y_test),
            np.abs(error).sum(),
            np.square(error).sum(),
            y_test.sum(),
            np.square(y_test).sum(),
        ))

        subsets_seen[key][subset] += 1

        if criteria is not None:
            selection_scores[key].append(criteria)

    for split_no, (train, test) in enumerate(splits, start=1):
        if len(test) < 2:
            raise ValueError(
                "At least two test recordings are needed for R²"
            )

        if protocol == "patient":
            assert set(groups[train]).isdisjoint(
                groups[test]
            )

        Xt, Xv = X[train], X[test]

        inner = make_inner_folds(
            Xt,
            groups[train],
            protocol,
            inner_count,
            seed + split_no,
        )

        for target in TARGETS:
            y = df[target].to_numpy(dtype=float)
            yt, yv = y[train], y[test]

            mean = yt.mean()

            remember(
                (target, "none", "Training mean"),
                yt,
                np.full(len(yt), mean),
                yv,
                np.full(len(yv), mean),
                (),
                None,
            )

            subsets, trees, criteria = select_feature_sets(
                Xt,
                yt,
                inner,
                scaling,
                seed + split_no,
            )

            fitted = {}

            for policy, subset in subsets.items():
                for name in MODELS:
                    cache_key = (subset, name)

                    if cache_key not in fitted:
                        if name == "CART":
                            model = trees[subset]
                        else:
                            model = fit_method(
                                name,
                                Xt[:, subset],
                                yt,
                                inner,
                                scaling,
                            )

                        fitted[cache_key] = (
                            model.predict(Xt[:, subset]),
                            model.predict(Xv[:, subset]),
                        )

                    train_pred, test_pred = fitted[cache_key]

                    remember(
                        (target, policy, name),
                        yt,
                        train_pred,
                        yv,
                        test_pred,
                        subset,
                        criteria[subset],
                    )

                    # Retain tracking data only for the paper's two plotted
                    # methods with its predefined six-feature subset.
                    if (
                        policy == "paper6"
                        and name in ("IRLS", "CART")
                    ):
                        for index, prediction in zip(
                            test,
                            test_pred,
                        ):
                            tracking[(target, name)][index].append(
                                float(prediction)
                            )

        if split_no == 1 or split_no % 25 == 0:
            print(
                f"{protocol}: completed split {split_no}",
                flush=True,
            )

    comparison = []

    for (target, policy, model), values in scores.items():
        a = np.asarray(values, dtype=float)

        n = a[:, 4].sum()

        sst = (
            a[:, 8].sum()
            - a[:, 7].sum() ** 2 / n
        )

        selected = selection_scores[
            (target, policy, model)
        ]

        ic = (
            np.mean(selected, axis=0)
            if selected
            else (np.nan, np.nan)
        )

        counts = subsets_seen[(target, policy, model)]

        comparison.append({
            "protocol": protocol,
            "target": target,
            "feature_set": policy,
            "model": model,
            "splits": len(a),

            "mean_train_mae": a[:, 0].mean(),
            "mean_test_mae": a[:, 1].mean(),
            "std_test_mae": (
                a[:, 1].std(ddof=1)
                if len(a) > 1
                else np.nan
            ),
            "mean_test_rmse": a[:, 2].mean(),
            "mean_test_r2": a[:, 3].mean(),

            "pooled_test_mae": a[:, 5].sum() / n,
            "pooled_test_rmse": np.sqrt(
                a[:, 6].sum() / n
            ),
            "pooled_test_r2": (
                1 - a[:, 6].sum() / sst
                if sst > 0
                else np.nan
            ),

            "mean_feature_count": (
                sum(
                    len(subset) * count
                    for subset, count in counts.items()
                )
                / len(a)
            ),

            "subset_cart_aic_mean": ic[0],
            "subset_cart_bic_mean": ic[1],

            "feature_subsets": json.dumps({
                (
                    ", ".join(
                        FEATURES[i] for i in subset
                    )
                    or "(none)"
                ): count
                for subset, count in sorted(counts.items())
            }),
        })

    predictions = []

    for (target, model), records in tracking.items():
        for index, values in enumerate(records):
            if not values:
                # Never fabricate predictions for untested rows.
                continue

            p5, p25, p75, p95 = np.percentile(
                values,
                [5, 25, 75, 95],
            )

            predictions.append({
                "protocol": protocol,
                "target": target,
                "feature_set": "paper6",
                "model": model,

                # Physical line in the original CSV, including its header.
                "recording_row": index + 2,

                "subject#": groups[index],
                "test_time": df.at[index, "test_time"],
                "actual_interpolated_updrs": df.at[index, target],

                "prediction_mean": np.mean(values),
                "prediction_std": np.std(values),
                "test_count": len(values),

                "p05": p5,
                "p25": p25,
                "p75": p75,
                "p95": p95,
            })

    return (
        pd.DataFrame(comparison),
        pd.DataFrame(predictions),
    )


def tracking_figure(
    predictions,
    destination,
    patient,
    protocol,
):
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    for ax, target in zip(axes, TARGETS):
        subset = predictions.loc[
            predictions["subject#"].eq(patient)
            & predictions.protocol.eq(protocol)
            & predictions.target.eq(target)
        ]

        if subset.empty:
            ax.text(
                0.5,
                0.5,
                "No held-out predictions for this patient",
                ha="center",
                transform=ax.transAxes,
            )

        else:
            actual = (
                subset.drop_duplicates("recording_row")
                .sort_values("test_time")
            )

            ax.plot(
                actual.test_time,
                actual.actual_interpolated_updrs,
                "k--",
                label="Supplied interpolated UPDRS",
            )

            for model, table in subset.groupby(
                "model",
                sort=False,
            ):
                table = table.sort_values([
                    "test_time",
                    "recording_row",
                ])

                line, = ax.plot(
                    table.test_time,
                    table.prediction_mean,
                    label=model,
                )

                ax.fill_between(
                    table.test_time,
                    table.p05,
                    table.p95,
                    color=line.get_color(),
                    alpha=0.15,
                )

            ax.legend(fontsize=8)

        ax.set_ylabel(target)

    axes[-1].set_xlabel("Days since recruitment")

    fig.suptitle(
        f"Patient {patient}; {protocol} evaluation; "
        "published six features\n"
        "Bands: repeated held-out prediction percentiles, "
        "not clinical confidence intervals"
    )

    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def make_report(df, comparison, args, figure_name):
    columns = [
        "protocol",
        "target",
        "feature_set",
        "model",
        "mean_test_mae",
        "std_test_mae",
        "pooled_test_mae",
        "pooled_test_r2",
    ]

    table = comparison[columns].round(4).astype(str)

    markdown = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    markdown += [
        "| " + " | ".join(row) + " |"
        for row in table.itertuples(
            index=False,
            name=None,
        )
    ]

    dataset_hash = hashlib.sha256(
        args.data.read_bytes()
    ).hexdigest()

    return "\n".join([
        "# Telemonitoring paper: reproduction-style experiment",
        "",

        (
            f"Data: `{args.data}`; recordings: {len(df)}; "
            f"patients: {df[GROUP].nunique()}."
        ),

        f"Dataset SHA-256: `{dataset_hash}`.",

        (
            f"Seed: {args.seed}; "
            f"recording repetitions: {args.repeats}; "
            f"patient folds: {args.group_folds}; "
            f"inner splits: {args.inner_folds}; "
            f"linear-model scaling: {args.scaling}."
        ),

        "",
        "## Explicit choices and limits",
        "",

        (
            "- Public UCI release: 5,875 recordings; paper: 5,923. "
            "Exact identity is unresolved."
        ),

        (
            "- No-intercept LS, IRLS and LASSO. IRLS counts "
            "initial LS as iteration one of 100."
        ),

        (
            "- Scaling defaults to none; maxabs is an optional, "
            "training-only alternative. The paper's plot normalization "
            "does not establish its regression scaling recipe."
        ),

        (
            "- LASSO alpha/alpha_max grid: "
            f"`{LASSO_RATIOS.tolist()}`."
        ),

        (
            f"- Sklearn CART ccp_alpha grid: `{list(CART_ALPHAS)}`; "
            "min_samples_leaf=1. This is not a claimed "
            "parameter-equivalent MATLAB implementation."
        ),

        (
            "- Inner selection: repeated 80/20 recording splits "
            "or grouped patient folds. Outer test labels never "
            "choose subsets, alpha or pruning."
        ),

        (
            "- LASSO-derived subsets are ranked using CART training "
            "RSS/reference_variance plus 2D or log(n)D. Reference "
            "variance is full-feature LS training MSE, shared across "
            "candidates. This explicit choice resolves an "
            "underspecified paper detail."
        ),

        (
            "- AIC/BIC columns describe CART-based subset selection, "
            "not each row's regression model."
        ),

        (
            "- Recording-level splitting can share patients across "
            "train/test. Patient-level splitting cannot."
        ),

        (
            "- Metrics are recording-weighted. Pooled scores use "
            "individual held-out predictions, not averaged prediction "
            "ensembles. Fold/repetition SD is not a clinical "
            "confidence interval."
        ),

        (
            "- Tracking CSV retains only paper6 IRLS/CART predictions, "
            "aggregated per recording. Percentile bands represent "
            "repeated-prediction variation; patient-wise rows "
            "have one prediction."
        ),

        (
            "- Targets are already interpolated. Tracking them "
            "is not future progression forecasting."
        ),

        (
            "- Other UPDRS target, patient ID, age, sex and "
            "test_time are never predictors."
        ),

        "",
        "## Results",
        "",
        *markdown,
        "",

        f"![Severity tracking]({figure_name})",

        "",

        (
            "Paper selected-six CART reference MAE: motor 5.95, "
            "total 7.52. These are historical reference values, "
            "not scores computed by this script."
        ),
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
    )

    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("recording", "patient"),
        default=["recording", "patient"],
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--group-folds",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--scaling",
        choices=("none", "maxabs"),
        default="none",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--plot-patient",
        type=int,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.repeats < 1 or args.inner_folds < 2:
        parser.error(
            "repeats must be positive; "
            "inner-folds must be at least two"
        )

    if len(set(args.protocols)) != len(args.protocols):
        parser.error("Do not repeat a protocol")

    regression = ROOT / "RESULTS/regression"
    reports = ROOT / "RESULTS/reports"

    paths = [
        regression / "paper_reproduction_comparison.csv",
        regression / "paper_reproduction_predictions.csv",
        reports / "paper_reproduction_report.md",
        reports / "paper_reproduction_tracking.png",
    ]

    if (
        not args.overwrite
        and any(path.exists() for path in paths)
    ):
        parser.error(
            "Reproduction outputs already exist; "
            "use --overwrite intentionally"
        )

    df = load_data(args.data)

    patient = (
        df[GROUP].min()
        if args.plot_patient is None
        else args.plot_patient
    )

    if patient not in set(df[GROUP]):
        parser.error(
            "plot-patient is not present in the dataset"
        )

    comparisons = []
    predictions = []

    for protocol in args.protocols:
        comparison, tracking = run_protocol(
            df,
            protocol,
            args.repeats,
            args.group_folds,
            args.inner_folds,
            args.scaling,
            args.seed,
        )

        comparisons.append(comparison)
        predictions.append(tracking)

    comparison = pd.concat(
        comparisons,
        ignore_index=True,
    )

    tracking = pd.concat(
        predictions,
        ignore_index=True,
    )

    baseline = (
        comparison.loc[
            comparison.model.eq("Training mean")
        ]
        .set_index(["protocol", "target"])
        ["pooled_test_mae"]
    )

    comparison["beats_mean_baseline"] = [
        (
            row.pooled_test_mae
            < baseline.loc[(row.protocol, row.target)]
        )
        for row in comparison.itertuples(index=False)
    ]

    regression.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(paths[0], index=False)
    tracking.to_csv(paths[1], index=False)

    plot_protocol = (
        "recording"
        if "recording" in args.protocols
        else "patient"
    )

    tracking_figure(
        tracking,
        paths[3],
        patient,
        plot_protocol,
    )

    paths[2].write_text(
        make_report(
            df,
            comparison,
            args,
            paths[3].name,
        ),
        encoding="utf-8",
    )

    print(
        comparison.drop(columns="feature_subsets")
        .round(4)
        .to_string(index=False)
    )

    for path in paths:
        print("Saved:", path)


if __name__ == "__main__":
    main()
