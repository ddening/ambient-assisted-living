import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from datetime import date
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import re
import copy

data_cleaned_dir = "../data/aruba/data_cleaned.csv"

df_aruba = pd.read_csv(
    data_cleaned_dir,
    delimiter=",",
    header=0,
    names=[
        "date",
        "time",
        "sensor_type",
        "sensor_status",
        "datetime",
    ],
)

df_aruba["datetime"] = pd.to_datetime(df_aruba["datetime"])
# df_aruba = df_aruba[:100]


def create_data_subset(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    df_subset = df[df["sensor_type"].str.match(pattern)]
    df_subset.loc[:, "date"] = pd.to_datetime(df_subset["date"])
    df_subset.loc[:, "datetime"] = pd.to_datetime(df_subset["datetime"], format="ISO8601")
    return df_subset


def activity_index(_df_motion_sensor: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_datetime = start_date
    end_datetime = end_date

    # Liste von Sensoren, für die der Aktivitätsindex berechnet werden soll (z. B. M001, M002)
    selected_sensors = []  # ["M001", "M003"]

    # Umwandeln der Zeitstempel in datetime-Objekte
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)

    # Umwandeln der Datums- und Zeitspalten in einen Datetime-Typ
    df_motion_sensor = _df_motion_sensor.copy()
    df_motion_sensor["datetime"] = pd.to_datetime(df_motion_sensor["datetime"])

    # Nur Zeilen mit 'ON' und 'OFF' Status
    df_motion_sensor_on_off = df_motion_sensor[df_motion_sensor["sensor_status"].isin(["ON", "OFF"])]

    # Filtere die Daten basierend auf dem angegebenen Zeitraum
    df_motion_sensor_on_off = df_motion_sensor_on_off[
        (df_motion_sensor_on_off["datetime"] >= start_datetime) & (df_motion_sensor_on_off["datetime"] <= end_datetime)
    ]

    # Sortieren nach Zeitstempel, falls noch nicht sortiert
    df_motion_sensor_on_off = df_motion_sensor_on_off.sort_values(by="datetime")

    # Filtere nur die gewünschten Sensoren
    if len(selected_sensors) != 0:
        df_motion_sensor_on_off = df_motion_sensor_on_off[
            df_motion_sensor_on_off["sensor_type"].isin(selected_sensors)
        ]

    # Liste der einzigartigen Sensoren
    unique_sensor_type = df_motion_sensor_on_off["sensor_type"].unique()
    unique_sensor_type = sorted(unique_sensor_type, key=lambda x: int(x[1:]))

    # Dictionary zum Speichern der Aktivitätsindizes für jeden Sensor
    activity_indices = {}
    datetime = {}

    # Berechnung des Aktivitätsindex für jeden Sensor
    for sensor in unique_sensor_type:
        # Filtere nur die Daten für den aktuellen Sensor
        df_sensor = df_motion_sensor_on_off[df_motion_sensor_on_off["sensor_type"] == sensor]

        # Berechnung der aktiven Zeiten (Dauer zwischen ON und OFF)
        active_times = []

        for i in range(1, len(df_sensor)):
            # Finde Paare von ON und OFF
            if df_sensor.iloc[i - 1]["sensor_status"] == "ON" and df_sensor.iloc[i]["sensor_status"] == "OFF":
                start_time = df_sensor.iloc[i - 1]["datetime"]
                end_time = df_sensor.iloc[i]["datetime"]
                active_duration = (end_time - start_time).total_seconds()  # Dauer in Sekunden
                active_times.append(active_duration)

        # Gesamtaktive Zeit
        total_active_time = np.sum(active_times)

        # Gesamtzeit des analysierten Zeitraums (vom ersten bis zum letzten Timestamp des Sensors)
        start_time_period = df_sensor["datetime"].min()
        end_time_period = df_sensor["datetime"].max()
        total_time_period = (end_time_period - start_time_period).total_seconds()

        # Berechnung des Aktivitätsindex für den Sensor
        activity_index = (total_active_time / total_time_period) * 100

        # Speichern des Aktivitätsindex im Dictionary
        activity_indices[sensor] = activity_index

    # Daten in einem DataFrame
    activity_index = [sensor_activity[1] for sensor_activity in activity_indices.items()]
    data = {"sensor": unique_sensor_type, "activity_index": activity_index}
    df_activity = pd.DataFrame(data)

    return df_activity, activity_indices


# Regex pattern for motion sensors (e.g., M001, M002, etc.)
pattern_motion_sensor = r"[M]{1}[0-9]{3,}"
pattern_temperature_sensor = r"[T]{1}[0-9]{3,}"
pattern_door_closure_sensor = r"[D]{1}[0-9]{3,}"

df_motion_sensor = create_data_subset(df_aruba, pattern_motion_sensor)

########################################
#######     SCHLAFANALYSE    ###########
########################################


def sleep_activity_index(df_input: pd.DataFrame, date_from, date_to, time_from, time_to) -> tuple:
    df_copy = df_input.copy()

    # Filter data for the specified range
    df_filtered = df_copy[
        (df_copy["date"] >= date_from)
        & (df_copy["date"] <= date_to)
        & (df_copy["time"] >= time_from)
        & (df_copy["time"] <= time_to)
    ]

    # Ensure data is sorted by sensor and datetime
    df_filtered = df_filtered.sort_values(by=["sensor_type", "datetime"])

    # Initialize an empty list to store results
    results = []

    # Group by each sensor and process
    for sensor, group in df_filtered.groupby("sensor_type"):
        # Reset index for easier access
        group = group.reset_index(drop=True)

        # Iterate through the group to find ON-OFF pairs
        for i in range(len(group) - 1):
            if group.loc[i, "sensor_status"] == "ON" and group.loc[i + 1, "sensor_status"] == "OFF":
                duration = (group.loc[i + 1, "datetime"] - group.loc[i, "datetime"]).total_seconds()
                results.append(
                    {
                        "sensor_type": sensor,
                        "activation_count": 1,
                        "total_active_duration": duration,
                        "datetime": group.loc[i, "datetime"],
                        "next_datetime": group.loc[i + 1, "datetime"],
                        "date": group.loc[i, "date"],
                        "time": group.loc[i, "time"],
                        "sensor_status": group.loc[i, "sensor_status"],
                    }
                )

    # Combine results into a summary DataFrame
    df_sleep_activity = pd.DataFrame(results)
    df_sleep_activity_index = pd.DataFrame()

    # Find the total time window for the sensor
    total_time_period = (df_filtered["datetime"].max() - df_filtered["datetime"].min()).total_seconds()

    # Aggregate by sensor_type
    if not df_sleep_activity.empty:
        sensor_summary = (
            df_sleep_activity.groupby("sensor_type")
            .agg(activation_count=("activation_count", "sum"), total_active_duration=("total_active_duration", "sum"))
            .reset_index()
        )

        # Add total_time_period as a constant column
        sensor_summary["total_time_period"] = total_time_period

        # Calculate activity_index
        sensor_summary["activity_index"] = (
            sensor_summary["total_active_duration"] / sensor_summary["total_time_period"]
        ) * 100

        df_sleep_activity_index = pd.DataFrame(sensor_summary)

    return (df_sleep_activity_index, df_sleep_activity)


def sleep_awake_ratio(
    df_bedroom_in_bed_sleep_activity_index: pd.DataFrame, df_bedroom_out_of_bed_sleep_activity_index: pd.DataFrame
):
    total_activity_in_bed = df_bedroom_in_bed_sleep_activity_index["activity_index"].sum()
    total_activity_out_of_bed = df_bedroom_out_of_bed_sleep_activity_index["activity_index"].sum()

    undefined_activity = abs(100 - (total_activity_in_bed + total_activity_out_of_bed))
    total_activity_in_bed = undefined_activity + total_activity_in_bed

    # Labels and data
    labels = ["Sleep Phase", "Awake Phase", "Undefined"]
    sleep_phase = total_activity_in_bed
    awake_phase = total_activity_out_of_bed
    data = [sleep_phase, awake_phase, undefined_activity]

    # Create the pie chart
    fig = px.pie(
        data,
        names=labels,
        values=data,
        title=f"Sleep vs Awake Phases",
    )

    return fig


app = Dash()

# App layout
app.layout = html.Div(
    [
        html.Div(children="Ambient Assisted Living Dashboard"),
        html.Hr(),
        dcc.DatePickerRange(
            id="date-picker-range-01",
            start_date=date(2010, 11, 4),
            end_date=date(2010, 11, 10),
            min_date_allowed=date(2010, 11, 4),
            max_date_allowed=date(2011, 6, 11),
            initial_visible_month=date(2010, 11, 4),
            display_format="YYYY-MM-DD",
            minimum_nights=0,
        ),
        # Time input fields for start and end times
        html.Div(
            [
                html.Label("Start Time (HH:MM):"),
                dcc.Input(
                    id="start-time-input-01",
                    type="text",
                    placeholder="00:00",
                    value="00:00",  # Default start time
                ),
            ]
        ),
        html.Div(
            [
                html.Label("End Time (HH:MM):"),
                dcc.Input(
                    id="end-time-input-01",
                    type="text",
                    placeholder="23:59",
                    value="23:59",  # Default end time
                ),
            ]
        ),
        dcc.Graph(figure={}, id="graph-sensor-activity-index"),
        ######################
        ######################
        dcc.DatePickerRange(
            id="date-picker-range-02",
            start_date=date(2010, 11, 15),
            end_date=date(2010, 11, 16),
            min_date_allowed=date(2010, 11, 4),
            max_date_allowed=date(2011, 6, 11),
            initial_visible_month=date(2010, 11, 15),
            display_format="YYYY-MM-DD",
            minimum_nights=0,
        ),
        # Time input fields for start and end times
        html.Div(
            [
                html.Label("Start Time (HH:MM):"),
                dcc.Input(
                    id="start-time-input-02",
                    type="text",
                    placeholder="00:00",
                    value="00:00",  # Default start time
                ),
            ]
        ),
        html.Div(
            [
                html.Label("End Time (HH:MM):"),
                dcc.Input(
                    id="end-time-input-02",
                    type="text",
                    placeholder="07:00",
                    value="07:00",  # Default end time
                ),
            ]
        ),
        dcc.Graph(figure={}, id="graph-sensor-activity-stacked"),
        ######################
        ######################
    ]
)


@callback(
    Output(component_id="graph-sensor-activity-index", component_property="figure"),
    Input("date-picker-range-01", "start_date"),
    Input("date-picker-range-01", "end_date"),
    Input("start-time-input-01", "value"),
    Input("end-time-input-01", "value"),
)
def update_sensor_activity_index(start_date, end_date, start_time, end_time) -> plt.figure:
    # Combine date and time inputs into a message
    if start_date and end_date:
        start = f"{start_date} {start_time}"
        end = f"{end_date} {end_time}"
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df_activity_index, _ = activity_index(df_motion_sensor, start, end)
        fig = px.histogram(df_activity_index, x="sensor", y="activity_index", histfunc="avg")
        return fig
    return None


@callback(
    Output(component_id="graph-sensor-activity-stacked", component_property="figure"),
    Input("date-picker-range-02", "start_date"),
    Input("date-picker-range-02", "end_date"),
    Input("start-time-input-02", "value"),
    Input("end-time-input-02", "value"),
)
def update_sleep_awake_ratio(start_date, end_date, start_time, end_time) -> plt.figure:
    # Combine date and time inputs into a message
    if start_date and end_date:
        start = f"{start_date} {start_time}"
        end = f"{end_date} {end_time}"
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        generate_sensor_list = lambda n, m: [f"M{str(i).zfill(3)}" for i in range(n, m + 1)]
        motion_sensor_list = generate_sensor_list(1, 31)

        filter_sensor_group = {
            "all": motion_sensor_list,
            "bedroom": ["M001", "M002", "M003", "M004", "M005", "M006", "M007"],
            "bedroom_in_bed": ["M002", "M003"],
            "bedroom_out_of_bed": ["M001", "M004", "M005", "M006", "M007"],
        }

        # Create a subset for motion sensors
        df_motion_sensor = create_data_subset(df_aruba, pattern_motion_sensor)
        df_motion_sensor = df_motion_sensor.reset_index(drop=True)

        df_motion_sensor.loc[:, "date"] = pd.to_datetime(df_motion_sensor["datetime"]).dt.date
        df_motion_sensor.loc[:, "time"] = (
            pd.to_datetime(df_motion_sensor["datetime"], format="%H:%M:%S.%f").dt.floor("s").dt.time
        )

        df_bedroom_in_bed = df_motion_sensor.loc[
            df_motion_sensor["sensor_type"].isin(filter_sensor_group["bedroom_in_bed"])
        ]

        df_bedroom_out_of_bed = df_motion_sensor.loc[
            df_motion_sensor["sensor_type"].isin(filter_sensor_group["bedroom_out_of_bed"])
        ]

        # Define the date and time ranges
        date_from = start.date()
        date_to = end.date()
        time_from = start.time()
        time_to = end.time()
        # print(date_from, date_to, time_from, time_to)

        # date_from = pd.to_datetime("2010-11-15").date()
        # date_to = pd.to_datetime("2010-11-16").date()
        # time_from = pd.to_datetime("00:00:00").time()
        # time_to = pd.to_datetime("07:00:00").time()

        df_bedroom_in_bed_sleep_activity_index, _ = sleep_activity_index(
            df_bedroom_in_bed, date_from, date_to, time_from, time_to
        )

        df_bedroom_out_of_bed_sleep_activity_index, _ = sleep_activity_index(
            df_bedroom_out_of_bed, date_from, date_to, time_from, time_to
        )

        fig = sleep_awake_ratio(df_bedroom_in_bed_sleep_activity_index, df_bedroom_out_of_bed_sleep_activity_index)

        return fig
    return None


if __name__ == "__main__":
    app.run(debug=True)
