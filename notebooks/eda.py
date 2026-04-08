import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    css_file="/home/dawid/Documents/python/f1/notebooks/theme.css",
)


@app.cell
def _():
    import pandas as pd
    import kagglehub
    import marimo as mo
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, kagglehub, mo, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    # Download and load data
    """)
    return


@app.cell
def _(Path, kagglehub):
    # Download latest version
    path = Path(
        kagglehub.dataset_download(
            "rohanrao/formula-1-world-championship-1950-2020"
        )
    )

    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(path, pd):
    circuits = pd.read_csv(path / "circuits.csv")

    constructor_results = pd.read_csv(path / "constructor_results.csv")
    constructor_standings = pd.read_csv(path / "constructor_standings.csv")
    constructors = pd.read_csv(path / "constructors.csv")

    driver_standings = pd.read_csv(path / "driver_standings.csv")
    drivers = pd.read_csv(path / "drivers.csv")

    lap_times = pd.read_csv(path / "lap_times.csv")

    pit_stops = pd.read_csv(path / "pit_stops.csv")

    qualifying = pd.read_csv(path / "qualifying.csv")

    races = pd.read_csv(path / "races.csv")
    race_results = pd.read_csv(path / "results.csv")

    seasons = pd.read_csv(path / "results.csv")

    sprint_results = pd.read_csv(path / "sprint_results.csv")

    status = pd.read_csv(path / "status.csv")
    return constructors, drivers, qualifying, race_results, races


@app.cell
def _(mo):
    mo.md(r"""
    # Display most important tables, clean and merge them
    """)
    return


@app.cell
def _(constructors):
    constructors
    return


@app.cell
def _(constructors, pd):
    def clean_constructors(df: pd.DataFrame) -> pd.DataFrame:
        to_drop = list(set(df.columns) - {"constructorId", "name"})
        return df.drop(columns=to_drop, inplace=False)


    def rename_name_column(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={"name": "constructor_name"})


    def change_constructor_name_type(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype({"constructor_name": "category"})


    constructors_cleaned = (
        constructors.pipe(clean_constructors)
        .pipe(rename_name_column)
        .pipe(change_constructor_name_type)
    )
    constructors_cleaned
    return (constructors_cleaned,)


@app.cell
def _(drivers):
    drivers
    return


@app.cell
def _(drivers, pd):
    def clean_drivers(df: pd.DataFrame) -> pd.DataFrame:
        to_drop = list(set(df.columns) - {"driverId", "forename", "surname"})
        return df.drop(columns=to_drop, inplace=False)


    def merge_columns_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(driver_name=df["forename"] + " " + df["surname"]).drop(
            columns=["forename", "surname"], inplace=False
        )


    def change_driver_name_type(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype({"driver_name": "category"})


    drivers_cleaned = (
        drivers.pipe(clean_drivers)
        .pipe(merge_columns_and_clean)
        .pipe(change_driver_name_type)
    )
    drivers_cleaned
    return (drivers_cleaned,)


@app.cell
def _(race_results):
    race_results
    return


@app.cell
def _(pd, race_results):
    def clean_race_results(df: pd.DataFrame) -> pd.DataFrame:
        to_drop = list(
            set(df.columns)
            - {"raceId", "driverId", "constructorId", "grid", "position"}
        )
        return df.drop(columns=to_drop, inplace=False)


    def change_position_type(df: pd.DataFrame) -> pd.DataFrame:
        if df["position"].dtype == object:
            return df.assign(
                position=df["position"].str.replace("\\N", "0").astype("int")
            )
        return df


    def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(
            columns={"grid": "start_position", "position": "race_result"}
        )


    race_results_cleaned = (
        race_results.pipe(clean_race_results)
        .pipe(change_position_type)
        .pipe(rename_columns)
    )
    race_results_cleaned
    return (race_results_cleaned,)


@app.cell
def _(races):
    races
    return


@app.cell
def _(pd, races):
    def clean_races(df: pd.DataFrame) -> pd.DataFrame:
        to_drop = list(set(df.columns) - {"raceId", "name", "date"})
        return df.drop(
            columns=to_drop,
            inplace=False,
        )


    def change_name_date_types(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(date=pd.to_datetime(df["date"])).astype(
            {"name": "category"}
        )


    def rename_race_name(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={"name": "race_name", "date": "race_date"})


    races_cleaned = (
        races.pipe(clean_races).pipe(change_name_date_types).pipe(rename_race_name)
    )
    races_cleaned
    return (races_cleaned,)


@app.cell
def _(qualifying):
    qualifying
    return


@app.cell
def _(pd, qualifying):
    def clean_qualifying(df: pd.DataFrame) -> pd.DataFrame:
        to_drop = list(set(df.columns) - {"raceId", "driverId", "position"})
        return df.drop(columns=to_drop, inplace=False)


    def rename_position(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={"position": "quali_result"})


    qualifying_cleaned = qualifying.pipe(clean_qualifying).pipe(rename_position)
    qualifying_cleaned
    return (qualifying_cleaned,)


@app.cell
def _(pd, qualifying_cleaned, race_results_cleaned, races_cleaned):
    def merge_races(df: pd.DataFrame, to_merge: pd.DataFrame) -> pd.DataFrame:
        return df.merge(right=to_merge, how="inner", on="raceId")


    def merge_qualifications(
        df: pd.DataFrame, to_merge: pd.DataFrame
    ) -> pd.DataFrame:
        return df.merge(right=to_merge, how="left", on=["driverId", "raceId"])


    full_race_weekend = race_results_cleaned.pipe(
        merge_races, to_merge=races_cleaned
    ).pipe(merge_qualifications, to_merge=qualifying_cleaned)
    full_race_weekend
    return (full_race_weekend,)


@app.cell
def _(constructors_cleaned, drivers_cleaned, full_race_weekend, pd):
    def merge_full_race_weekend(
        df: pd.DataFrame, to_merge: pd.DataFrame
    ) -> pd.DataFrame:
        return df.merge(right=to_merge, how="right", on="driverId")


    def merge_constructors_cleaned(
        df: pd.DataFrame, to_merge: pd.DataFrame
    ) -> pd.DataFrame:
        return df.merge(right=to_merge, how="left", on="constructorId")


    def sort_and_select_dates(
        df: pd.DataFrame, start_date: str = "2014-01-01", end_date: str = ""
    ) -> pd.DataFrame:
        sorted_df = df.sort_values(by="race_date", ascending=True, inplace=False)
        if end_date:
            return sorted_df[
                (sorted_df["race_date"] >= pd.to_datetime(start_date))
                & (sorted_df["race_date"] <= pd.to_datetime(end_date))
            ]
        return sorted_df[sorted_df["race_date"] >= pd.to_datetime(start_date)]


    def drop_ids(df: pd.DataFrame) -> pd.DataFrame:
        to_drop = ["driverId", "raceId", "constructorId"]
        return df.drop(columns=to_drop, inplace=False)


    def fix_quali_result_column(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            quali_result=df["quali_result"]
            .fillna(df["start_position"])
            .astype("int")
        )


    def remove_dates(df: pd.DataFrame) -> pd.DataFrame:
        to_drop = ["race_date"]
        return df.drop(columns=to_drop, inplace=False)


    def fix_categories(df: pd.DataFrame) -> pd.DataFrame:
        df["driver_name"] = df["driver_name"].cat.remove_unused_categories()
        df["constructor_name"] = df[
            "constructor_name"
        ].cat.remove_unused_categories()
        df["race_name"] = df["race_name"].cat.remove_unused_categories()
        return df


    driver_performance = (
        drivers_cleaned.pipe(merge_full_race_weekend, to_merge=full_race_weekend)
        .pipe(merge_constructors_cleaned, to_merge=constructors_cleaned)
        .pipe(drop_ids)
        .pipe(sort_and_select_dates)
        .pipe(fix_quali_result_column)
        .pipe(fix_categories)
        # .pipe(remove_dates)
    )
    driver_performance
    return (driver_performance,)


@app.cell
def _(driver_performance):
    # def replace_zeros(series: pd.Series) -> pd.Series:
    #     value = driver_count_per_race[
    #         driver_count_per_race["race_date"] == series.race_date
    #     ]["driver_name"]
    #     out_series = series.replace(to_replace=0, value=value.values[0])
    #     return out_series

    driver_count_per_race = driver_performance.groupby(by="race_date")[
        "driver_name"
    ].transform("count")
    driver_performance_2 = driver_performance.copy(deep=True)

    mask = driver_performance_2["race_result"] == 0
    driver_performance_2.loc[mask, "race_result"] = driver_count_per_race[mask]


    mask_2 = driver_performance_2["quali_result"] == 0
    driver_performance_2.loc[mask_2, "quali_result"] = driver_count_per_race[
        mask_2
    ]


    mask_3 = driver_performance_2["start_position"] == 0
    driver_performance_2.loc[mask_3, "start_position"] = driver_count_per_race[
        mask_3
    ]

    driver_performance_2
    return (driver_performance_2,)


@app.cell
def _(driver_performance_2):
    driver_performance_2["driver_rolling_avg"] = (
        driver_performance_2.groupby("driver_name", observed=False)["race_result"]
        .rolling(window=3, min_periods=1)
        .mean()
        .round(decimals=2)
        .reset_index(level=0, drop=True)
    )
    driver_performance_2
    return


@app.cell
def _(driver_performance_2):
    tmp = (
        driver_performance_2.groupby(
            ["race_date", "constructor_name"], observed=False
        )["race_result"]
        .mean()
        .fillna(0)
        .reset_index(name="team_race_result")
        .sort_values(by=["race_date", "constructor_name"])
    )
    tmp["constructor_rolling_avg"] = (
        tmp.groupby(["constructor_name"], observed=False)["team_race_result"]
        .rolling(window=3, min_periods=1)
        .mean()
        .round(decimals=2)
        .reset_index(level=0, drop=True)
    )
    tmp
    return (tmp,)


@app.cell
def _(driver_performance_2, tmp):
    driver_performance_3 = driver_performance_2.merge(
        right=tmp[["race_date", "constructor_name", "constructor_rolling_avg"]],
        on=["race_date", "constructor_name"],
    )
    driver_performance_3
    return (driver_performance_3,)


@app.cell
def _(driver_performance_3):
    x = driver_performance_3.groupby(["driver_name", "race_name"], observed=False)[
        "race_date"
    ].count()
    x
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Explore, plot, and analyze data
    """)
    return


@app.cell
def _(driver_performance_3):
    dp = driver_performance_3
    return (dp,)


@app.cell
def _(dp, sns):
    sns.boxplot(dp, x="start_position", y="race_result")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Let's plot and see, whether the place at the start of the grand prix determine the race result for a driver.
    """)
    return


@app.cell
def _(pd, sns):
    def heatmap(x: pd.Series, y: pd.Series):
        crosstab = pd.crosstab(y, x)
        ax = sns.heatmap(crosstab, cmap="viridis", annot=False)
        return ax.figure

    return (heatmap,)


@app.cell
def _(dp, heatmap):
    heatmap(dp["start_position"], dp["race_result"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Why are we plotting race_result against the start_position and then once again against the qualifying result? The second heat map illustrates better the tempo of a driver and a car, while the first shows us the starting positions after qualifying and grid penalties e.g. for blocking another driver or car changes like PU (power unit).
    """)
    return


@app.cell
def _(dp, heatmap):
    heatmap(dp["quali_result"], dp["race_result"])
    return


@app.cell
def _(dp):
    dp.corr(method="pearson", numeric_only=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### We can clearly see that the higher we qualify and start the race, the higher are chances that we will finish the race or even win it. While being in the middel of a F1 pack makes it more probable that driver will crash with others.
    """)
    return


@app.cell
def _(dp, heatmap):
    heatmap(dp["constructor_name"], dp["race_result"])
    return


@app.cell
def _(dp, heatmap):
    heatmap(dp["constructor_name"], dp["quali_result"])
    return


@app.cell
def _(dp, plt, sns):
    constructor_wins = (
        dp[["constructor_name", "race_result"]][dp["race_result"] == 1]
        .groupby("constructor_name", observed=True, as_index=False)
        .count()
    )

    plt.subplots(figsize=(20, 10))
    sns.barplot(
        constructor_wins,
        x="constructor_name",
        y="race_result",
        order=constructor_wins.sort_values(by="race_result", ascending=False)[
            "constructor_name"
        ],
    )
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### We can see that three teams are way ahead of others, and those are Mercedes, Ferrari and RedBull.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary
    ### The cutoff date for the data is 2024-12-08(YYYY-MM-DD), so it gives us the whole 2025 season to then test models upon, maybe we will be able to accurately predict, who got championship this year(Lando Norris).
    ### Qualifying position and start position are very strong predictors, as to whether driver will score good on the Saturday, or even finish the race. Because the data show, that the further from the pole position we start, the higher the chance we will not finish a race. The most endangered drivers are those in the so called midfield.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### As we have some information about our our data, I think that it is a good time to start preparing input data for models, and create some additional features, that would for example represent drivers/constructors recent form from 5 last races.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Models

    ### Now we will try out a few different models on our data, and see what we get in return. If they behave quite well on the data, we will try and make some additional data for the models to learn on, to make them even better. If it turns out that we don't get the desired results on many different models, we will try to rething data that we them with.
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor

    return


@app.cell
def _():
    # dp_for_train = pd.concat(
    #     [dp, pd.get_dummies(dp["constructor_name"])], axis=1
    # ).drop(columns=["driver_name", "race_name", "constructor_name"])
    # dp_for_train
    # def prepare_data_for_train_test(df: pd.DataFrame) -> pd.DataFrame:
    #     return df.drop(
    #         columns=["driver_name", "race_name", "race_date", "constructor_name"],
    #         inplace=False,
    #     )
    #     return


    # dp_for_training = prepare_data_for_train_test(dp)
    # dp_for_training
    return


@app.cell
def _(dp):
    dp
    return


@app.cell
def _(dp, pd):
    def train_test_split(
        df: pd.DataFrame, train_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_index = int(train_size * len(dp.index))

        start_date = df.iloc[[train_index]]["race_date"].values[0]

        race_dates = tuple(df.iloc[train_index:]["race_date"])

        for race_date in race_dates:
            if start_date != race_date:
                break
            train_index += 1

        train_index += 1
        X_train, y_train = (
            df.iloc[:train_index],
            df.iloc[:train_index]["race_result"],
        )

        X_test, y_test = (
            df.iloc[train_index:],
            df.iloc[train_index:]["race_result"],
        )
        return X_train, X_test, y_train, y_test


    X_train, X_test, y_train, y_test = train_test_split(dp, train_size=0.8)
    return X_test, X_train, y_test


@app.cell
def _(X_train):
    race_dates = X_train["race_date"]

    X_train.drop(
        columns=[
            "race_date",
            "driver_name",
            "race_result",
            "race_name",
            "constructor_name",
        ],
        inplace=False,
    )
    race_dates
    return (race_dates,)


@app.cell
def _(np, pd):
    class GroupTimeSeriesSplit:
        def __init__(
            self, race_dates, n_splits: int = 5, max_train_size: int | None = None
        ) -> None:
            self.race_dates = np.array(race_dates)
            self.n_splits = n_splits
            self.max_train_size = max_train_size

        def split(self, X: pd.DataFrame, y=None, groups=None):
            groups = np.unique(self.race_dates)
            # print(groups)

            # print(f"{type(self.race_dates)=}, {type(groups)=}")

            fold_size = len(groups) // (self.n_splits + 1)

            for i in range(0, self.n_splits):
                train_end = (i + 1) * fold_size
                test_end = train_end + fold_size
                train_groups = X[:train_end]
                test_groups = X[train_end:test_end]
                # print(
                #     f"{len(groups)=}\n{fold_size=}\n{train_end=}\n{test_end=}\n{i=}\n\n"
                # )
                train_idx = np.nonzero(np.isin(self.race_dates, train_groups))
                test_idx = np.flatnonzero(np.isin(self.race_dates, train_groups))

                if i == 0:
                    print(f"{len(self.race_dates)=}")
                    print(train_idx)

                # print(f"{len(train_idx)=}\n{len(test_idx)=}\n\n")

                yield train_idx, test_idx

        def get_n_splits(self, X, y=None, groups=None) -> int:
            return self.n_splits

    return (GroupTimeSeriesSplit,)


@app.cell
def _(GroupTimeSeriesSplit, X_train, race_dates):
    splitter = GroupTimeSeriesSplit(race_dates=race_dates, n_splits=6)
    output = splitter.split(X_train)
    next(output)
    return


@app.cell
def _():
    # search = GridSearchCV(
    #     estimator=RandomForestRegressor(random_state=42),
    #     param_grid={
    #         "n_estimators": [100, 200, 300],
    #         "max_depth": [None, 10, 50, 100],
    #         "min_samples_leaf": [1, 2, 5, 10],
    #     },
    #     cv=GroupTimeSeriesSplit(race_dates=race_dates, n_splits=6),
    # )
    # search.fit(X_train, y_train)
    return


@app.cell
def _(X_test, y_test):
    start = 20
    stop = 30
    test_sample, test_sample_ans = X_test.iloc[start:stop], y_test.iloc[start:stop]
    print(f"{test_sample=}\n{test_sample_ans=}")
    return test_sample, test_sample_ans


@app.cell
def _(search, test_sample, test_sample_ans):
    best_estimator = search.best_estimator_
    test_ans = best_estimator.predict(test_sample)
    print(f"{test_ans - test_sample_ans}")
    return


if __name__ == "__main__":
    app.run()
