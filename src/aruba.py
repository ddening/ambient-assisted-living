from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import pandas as pd
import seaborn as sns
import re

data_dir = "../data/aruba/data.csv"

df_aruba = pd.read_csv(
    data_dir,
    delim_whitespace=True,
    names=[
        "date",
        "time",
        "sensor_type",
        "sensor_status",
        "meta",
        "meta_begin_end",
    ],
)

pattern_motion_sensor = r"[M]{1}[0-9]{3,}"
pattern_temperature_sensor = r"[T]{1}[0-9]{3,}"
pattern_door_closure_sensor = r"[D]{1}[0-9]{3,}"

# Create subsets for each sensor type
df_motion_sensor = df_aruba[
    df_aruba["sensor_type"].str.match(pattern_motion_sensor)
]
df_temperature_sensor = df_aruba[
    df_aruba["sensor_type"].str.match(pattern_temperature_sensor)
]
df_door_closure_sensor = df_aruba[
    df_aruba["sensor_type"].str.match(pattern_door_closure_sensor)
]

# display(df_temperature_sensor)
# display(df_temperature_sensor["date"])


# Konvertiere Datum und Uhrzeit in eine einzige Spalte mit einem Datetime-Objekt
df_temperature_sensor["datetime"] = pd.to_datetime(
    df_temperature_sensor["date"] + " " + df_temperature_sensor["time"]
)
display(df_temperature_sensor)

# Sortiere die Daten nach Datum und Uhrzeit
df_temperature_sensor = df_temperature_sensor.sort_values(by="datetime")

# Konvertiere die sensor_status-Spalte in numerische Werte
df_temperature_sensor["sensor_status"] = pd.to_numeric(
    df_temperature_sensor["sensor_status"], errors="coerce"
)

# Erstelle ein Diagramm für jeden Temperatursensor
unique_sensors = df_temperature_sensor["sensor_type"].unique()

# Array basierend auf den numerischen Teilen der IDs sortieren
sorted_sensors = sorted(unique_sensors, key=lambda x: int(x[1:]))
print(sorted_sensors)

# Interaktive Darstellung mit CheckButtons
fig, ax = plt.subplots(figsize=(12, 6))

# Linien für jeden Sensor speichern
lines = {}
for sensor_id in sorted_sensors:
    sensor_data = df_temperature_sensor[
        df_temperature_sensor["sensor_type"] == sensor_id
    ]
    (line,) = ax.plot(
        sensor_data["datetime"], sensor_data["sensor_status"], label=sensor_id
    )
    lines[sensor_id] = line

ax.set_title("Temperaturdaten")
ax.set_xlabel("Zeit")
ax.set_ylabel("Temperatur")
ax.legend()
ax.grid(True)

# CheckButtons hinzufügen
rax = plt.axes(
    [0.02, 0.4, 0.15, 0.4]
)  # Position der CheckButtons [x, y, Breite, Höhe]
visibility = [True] * len(sorted_sensors)  # Alle Graphen starten sichtbar
check = CheckButtons(rax, sorted_sensors, visibility)


# Funktion zum Aktualisieren der Sichtbarkeit
def toggle_visibility(label):
    line = lines[label]
    line.set_visible(not line.get_visible())
    plt.draw()


check.on_clicked(toggle_visibility)
plt.show()
