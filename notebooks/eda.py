import marimo

__generated_with = "0.18.4"
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
    return kagglehub, mo, pd, sns


@app.cell
def _(mo):
    mo.md(r"""
    # Download and load data
    """)
    return


@app.cell
def _(kagglehub):
    # Download latest version
    path = kagglehub.dataset_download(
        "rohanrao/formula-1-world-championship-1950-2020"
    )

    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(path, pd):
    circuits = pd.read_csv(path + "/circuits.csv")

    constructor_results = pd.read_csv(path + "/constructor_results.csv")
    constructor_standings = pd.read_csv(path + "/constructor_standings.csv")
    constructors = pd.read_csv(path + "/constructors.csv")

    driver_standings = pd.read_csv(path + "/driver_standings.csv")
    drivers = pd.read_csv(path + "/drivers.csv")

    lap_times = pd.read_csv(path + "/lap_times.csv")

    pit_stops = pd.read_csv(path + "/pit_stops.csv")

    qualifying = pd.read_csv(path + "/qualifying.csv")

    races = pd.read_csv(path + "/races.csv")
    race_results = pd.read_csv(path + "/results.csv")

    seasons = pd.read_csv(path + "/results.csv")

    sprint_results = pd.read_csv(path + "/sprint_results.csv")

    status = pd.read_csv(path + "/status.csv")
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
    def clean_constructors(df: pd.DataFrame):
        to_drop = list(set(df.columns) - {"constructorId", "name"})
        return df.drop(columns=to_drop, inplace=False)


    def rename_name_column(df: pd.DataFrame):
        return df.rename(columns={"name": "constructor_name"})


    def change_constructor_name_type(df: pd.DataFrame):
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
    def clean_drivers(df: pd.DataFrame):
        to_drop = list(set(df.columns) - {"driverId", "forename", "surname"})
        return df.drop(columns=to_drop, inplace=False)


    def merge_columns_and_clean(df: pd.DataFrame):
        return df.assign(driver_name=df["forename"] + " " + df["surname"]).drop(
            columns=["forename", "surname"], inplace=False
        )


    def change_driver_name_type(df: pd.DataFrame):
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
    def clean_race_results(df: pd.DataFrame):
        to_drop = list(
            set(df.columns)
            - {"raceId", "driverId", "constructorId", "grid", "position"}
        )
        return df.drop(columns=to_drop, inplace=False)


    def change_position_type(df: pd.DataFrame):
        if df["position"].dtype == object:
            return df.assign(
                position=df["position"].str.replace("\\N", "0").astype("int")
            )
        return df


    def rename_columns(df: pd.DataFrame):
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
    def clean_races(df: pd.DataFrame):
        to_drop = list(set(df.columns) - {"raceId", "name", "date"})
        return df.drop(
            columns=to_drop,
            inplace=False,
        )


    def change_name_date_types(df: pd.DataFrame):
        return df.assign(date=pd.to_datetime(df["date"])).astype(
            {"name": "category"}
        )


    def rename_race_name(df: pd.DataFrame):
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
    def clean_qualifying(df: pd.DataFrame):
        to_drop = list(set(df.columns) - {"raceId", "driverId", "position"})
        return df.drop(columns=to_drop, inplace=False)


    def rename_position(df: pd.DataFrame):
        return df.rename(columns={"position": "quali_result"})


    qualifying_cleaned = qualifying.pipe(clean_qualifying).pipe(rename_position)
    qualifying_cleaned
    return (qualifying_cleaned,)


@app.cell
def _(pd, qualifying_cleaned, race_results_cleaned, races_cleaned):
    def merge_races(df: pd.DataFrame, to_merge: pd.DataFrame):
        return df.merge(right=to_merge, how="inner", on="raceId")


    def merge_qualifications(df: pd.DataFrame, to_merge: pd.DataFrame):
        return df.merge(right=to_merge, how="left", on=["driverId", "raceId"])


    full_race_weekend = race_results_cleaned.pipe(
        merge_races, to_merge=races_cleaned
    ).pipe(merge_qualifications, to_merge=qualifying_cleaned)
    full_race_weekend
    return (full_race_weekend,)


@app.cell
def _(constructors_cleaned, drivers_cleaned, full_race_weekend, pd):
    def merge_full_race_weekend(df: pd.DataFrame, to_merge: pd.DataFrame):
        return df.merge(right=to_merge, how="right", on="driverId")


    def merge_constructors_cleaned(df: pd.DataFrame, to_merge: pd.DataFrame):
        return df.merge(right=to_merge, how="left", on="constructorId")


    def sort_and_select_dates(
        df: pd.DataFrame, start_date: str = "2014-01-01", end_date: str = ""
    ):
        sorted_df = df.sort_values(by="race_date", ascending=True, inplace=False)
        if end_date:
            return sorted_df[
                (sorted_df["race_date"] >= pd.to_datetime(start_date))
                & (sorted_df["race_date"] <= pd.to_datetime(end_date))
            ]
        return sorted_df[sorted_df["race_date"] >= pd.to_datetime(start_date)]


    def drop_ids(df: pd.DataFrame):
        to_drop = ["driverId", "raceId", "constructorId"]
        return df.drop(columns=to_drop, inplace=False)


    def fix_quali_result_column(df: pd.DataFrame):
        return df.assign(
            quali_result=df["quali_result"].fillna(df["start_position"])
        ).astype({"quali_result": "int"})


    driver_performance = (
        drivers_cleaned.pipe(merge_full_race_weekend, to_merge=full_race_weekend)
        .pipe(merge_constructors_cleaned, to_merge=constructors_cleaned)
        .pipe(drop_ids)
        .pipe(sort_and_select_dates)
        .pipe(fix_quali_result_column)
    )
    driver_performance
    return (driver_performance,)


@app.cell
def _(mo):
    mo.md(r"""
    # Explore, plot, and analyze data
    """)
    return


@app.cell
def _(driver_performance, sns):
    sns.boxplot(driver_performance, x="start_position", y="race_result")
    return


@app.cell
def _(driver_performance, pd, sns):
    start_result_heatmap = pd.crosstab(
        driver_performance["race_result"], driver_performance["start_position"]
    )
    sns.heatmap(start_result_heatmap, cmap="viridis", annot=False)
    return


@app.cell
def _(driver_performance, pd, sns):
    quali_result_heatmap = pd.crosstab(
        driver_performance["race_result"], driver_performance["quali_result"]
    )
    sns.heatmap(quali_result_heatmap, cmap="viridis", annot=False)
    return


if __name__ == "__main__":
    app.run()
