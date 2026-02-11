from pathlib import Path
import pandas as pd
from google.transit import gtfs_realtime_pb2

# Real time data is in Protocol Buffers format
# --- Paths ---
pb_dir = Path("data/raw/gtfs_rt/trip_updates")
out_dir = Path("data/interim/gtfs_rt")
out_dir.mkdir(parents=True, exist_ok=True)
# ensures the output folder exists so the script won’t crash when saving.

# --- Pick latest .pb file ---
pb_files = sorted(pb_dir.glob("trip_updates_*.pb"))
if not pb_files:
    raise FileNotFoundError(f"No .pb files found in {pb_dir}. Run the download script first.")

latest_pb = pb_files[-1]
# selects the most recent snapshot.

# --- Read + decode ---
feed = gtfs_realtime_pb2.FeedMessage()
feed.ParseFromString(latest_pb.read_bytes())

rows = []
feed_ts = feed.header.timestamp  # unix seconds

for ent in feed.entity:
    if not ent.HasField("trip_update"):
        continue
# skip entities without trip updates

    tu = ent.trip_update
    trip = tu.trip

    trip_id = getattr(trip, "trip_id", None)
    route_id = getattr(trip, "route_id", None)
    direction_id = getattr(trip, "direction_id", None)
    start_date = getattr(trip, "start_date", None)
    start_time = getattr(trip, "start_time", None)

    # Each stop_time_update may have arrival/departure times + delay seconds
    for stu in tu.stop_time_update:
        stop_id = getattr(stu, "stop_id", None)
        stop_sequence = getattr(stu, "stop_sequence", None)
# loop through stop time updates and extract relevant fields

        # delay can exist on arrival or departure
        arr_delay = stu.arrival.delay if stu.HasField("arrival") and stu.arrival.HasField("delay") else None
        dep_delay = stu.departure.delay if stu.HasField("departure") and stu.departure.HasField("delay") else None

        # prefer arrival_delay, fallback to departure_delay
        delay_sec = arr_delay if arr_delay is not None else dep_delay

        # predicted actual times (epoch seconds) if available
        arr_time = stu.arrival.time if stu.HasField("arrival") and stu.arrival.HasField("time") else None
        dep_time = stu.departure.time if stu.HasField("departure") and stu.departure.HasField("time") else None
# the step above extracts delay information from GTFS-Realtime stop updates
# Standardizes delay into a single variable (delay_sec)
# Extracts predicted actual arrival/departure times, if available
# Handles missing fields gracefully
# Basiscally, look into each stop up date and asssign if the bus is delayed or not for arrival and departure

# the step below reformats the extracted data into a list of dictionaries for DataFrame conversion
        rows.append(
            {
                "feed_timestamp": feed_ts,
                "entity_id": ent.id,
                "trip_id": trip_id,
                "route_id": route_id,
                "direction_id": direction_id,
                "start_date": start_date,
                "start_time": start_time,
                "stop_id": stop_id,
                "stop_sequence": stop_sequence,
                "delay_sec": delay_sec,
                "arrival_time_epoch": arr_time,
                "departure_time_epoch": dep_time,
            }
        )

df = pd.DataFrame(rows)

# Convert delay to minutes + label
df["delay_min"] = df["delay_sec"] / 60.0
df["delay_10plus"] = df["delay_min"] >= 10  # Changed from 15 to 10 minutes

out_path = out_dir / f"trip_updates_parsed_{feed_ts}.csv"
df.to_csv(out_path, index=False)

print("Parsed latest file:", latest_pb.name)
print("Rows:", len(df))
print("Saved CSV:", out_path)
print(df[["route_id", "trip_id", "stop_id", "delay_min", "delay_10plus"]].head(10))
