import json
from datetime import datetime
from functools import reduce
from pathlib import Path

import click
import numpy as np
import pandas as pd

from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor


def arrays_to_datetime(days_since_2000, seconds_since_midnight):
    """Convert GPS9 days/seconds to datetime with Unix nanosecond epoch."""
    # Reference date: January 1, 2000
    reference_date = datetime(2000, 1, 1)

    # Convert to pandas datetime
    datetime_series = (
        pd.to_datetime(reference_date)
        + pd.to_timedelta(days_since_2000, unit="D")
        + pd.to_timedelta(seconds_since_midnight, unit="s")
    )

    return datetime_series


def convert_to_absolute_timestamps(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Convert relative timestamps to absolute Unix epoch nanoseconds using GPS9 reference.

    Args:
        dataframes: Dictionary of sensor DataFrames with relative timestamps

    Returns:
        Dictionary of DataFrames with absolute timestamps (modifies in place)

    Raises:
        ValueError: If GPS9 data is not available for reference

    """
    gps_reference_time = None

    # Find GPS9 data to get reference timestamp
    if "gps9" in dataframes and not dataframes["gps9"].empty:
        gps_df = dataframes["gps9"]
        if "gps9_datetime_unix_ns" in gps_df.columns:
            # Use the first GPS datetime as reference
            first_gps_time_ns = gps_df["gps9_datetime_unix_ns"].iloc[0]
            first_gps_timestamp = gps_df["timestamp"].iloc[0]
            gps_reference_time = first_gps_time_ns - (first_gps_timestamp * 1e9)  # Convert to ns
            print(f"📅 Using GPS9 reference time: {pd.to_datetime(first_gps_time_ns, unit='ns')}")

    # Fallback to GPS5 if GPS9 not available
    elif "gps5" in dataframes and not dataframes["gps5"].empty:
        print(
            "⚠️  GPS9 not available, cannot compute absolute timestamps without GPS datetime reference"
        )
        gps_reference_time = None

    if gps_reference_time is not None:
        # Convert all timestamps to absolute Unix epoch nanoseconds
        for sensor_name, df in dataframes.items():
            if not df.empty and "timestamp" in df.columns:
                # Convert relative timestamps to absolute Unix epoch nanoseconds
                df["timestamp"] = (df["timestamp"] * 1e9 + gps_reference_time).astype("int64")
                print(f"✅ Converted {sensor_name} timestamps to absolute Unix epoch nanoseconds")
    else:
        raise ValueError("Cannot convert to absolute time: No GPS9 data with datetime available")

    return dataframes


def create_sensor_df(
    data: np.ndarray, timestamps: np.ndarray, sensor_name: str, axis_names: list[str]
) -> pd.DataFrame:
    """Helper function to create DataFrame with data and timestamps."""
    if len(data) == 0:
        # Return empty DataFrame with proper columns if no data
        columns = ["timestamp"] + [f"{sensor_name}_{axis}" for axis in axis_names]
        return pd.DataFrame(columns=columns)

    # Create DataFrame with sensor data
    if data.ndim == 1:
        # 1D data (single value per timestamp)
        df = pd.DataFrame({f"{sensor_name}_value": data})
    else:
        # Multi-dimensional data (e.g., X, Y, Z axes)
        sensor_columns = {
            f"{sensor_name}_{axis}": data[:, i]
            for i, axis in enumerate(axis_names[: data.shape[1]])
        }
        df = pd.DataFrame(sensor_columns)

    # Add timestamp column
    df["timestamp"] = timestamps

    # Special handling for GPS9 data - add computed datetime
    if sensor_name == "gps9" and len(data) > 0 and data.shape[1] >= 7:
        try:
            # GPS9 structure: lat, lon, alt, speed_2d, speed_3d, days_since_2000, secs_since_midnight_ms, dop, fix
            days_since_2000 = data[:, 5]  # Column 5: days since 2000
            seconds_since_midnight = data[:, 6] / 1000.0  # Column 6: seconds with ms precision

            # Compute datetime using your method
            gps_datetime = arrays_to_datetime(days_since_2000, seconds_since_midnight)

            # Add as Unix nanosecond epoch
            df["gps9_datetime_unix_ns"] = gps_datetime.astype("int64")

            print(f"✅ Added GPS9 datetime columns for {len(df)} GPS samples")

        except Exception as e:
            print(f"⚠️  Failed to compute GPS9 datetime: {e}")

    # Reorder columns to put timestamp first
    cols = ["timestamp"] + [col for col in df.columns if col != "timestamp"]
    return df[cols]


def load_sensor_configs(config_path: Path) -> dict[str, dict]:
    """Load sensor configurations from JSON file with basic validation."""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Sensor config not found at {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in sensor config: {config_path}") from exc

    # Basic validation
    if not isinstance(data, dict):
        raise ValueError("Sensor config must be a mapping of FOURCC -> config dict")
    for fourcc, cfg in data.items():
        if not isinstance(fourcc, str) or len(fourcc) != 4:
            raise ValueError(f"Invalid sensor key '{fourcc}': must be a 4-character FOURCC")
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid config for {fourcc}: must be a dict")
        if "name" not in cfg or "axis_names" not in cfg:
            raise ValueError(f"Config for {fourcc} missing required fields 'name'/'axis_names'")
        if not isinstance(cfg["axis_names"], list):
            raise ValueError(f"Config for {fourcc} 'axis_names' must be a list")

    return data


def make_dfs(
    filepath: Path, config_path: Path, absolute_time: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Extract telemetry data from GoPro MP4 file and return as separate DataFrames.

    Args:
        filepath: Path to the MP4 file.
        config_path: Filesystem path to the sensors JSON configuration that
            defines available FOURCC keys, axis names, and descriptions.
        absolute_time: If True, converts timestamps to absolute Unix epoch nanoseconds
            using the first GPS timestamp as reference.

    Returns:
        Dictionary of DataFrames, one for each sensor type

    """
    extractor = GoProTelemetryExtractor(filepath)
    extractor.open_source()

    # Determine which channels to extract
    sensor_configs = load_sensor_configs(config_path)
    # Validate requested channels
    sensors = list(sensor_configs.keys())
    invalid_channels = [ch for ch in sensors if ch not in sensors]
    if invalid_channels:
        print(f"Warning: Unknown sensor types: {invalid_channels}")
        print(f"Available sensors: {sensors}")

    channels_to_extract = [ch for ch in sensors if ch in sensors]

    # Extract sensor data dynamically
    dataframes = {}

    for sensor_type in channels_to_extract:
        try:
            # Extract data for this sensor
            data, timestamps = extractor.extract_data(sensor_type)

            # Get sensor configuration
            config = sensor_configs[sensor_type]

            # Create DataFrame if data exists
            if len(data) > 0:
                df = create_sensor_df(data, timestamps, config["name"], config["axis_names"])
                dataframes[config["name"]] = df
                print(
                    f"✅ Created {config['name']} DataFrame: {len(df)} rows - {config['description']}"
                )
            else:
                print(f"⚠️  No data found for {sensor_type} - {config['description']}")

        except Exception as e:
            print(f"❌ Error extracting {sensor_type}: {e}")

    extractor.close_source()

    # Convert to absolute timestamps if requested
    if absolute_time:
        try:
            dataframes = convert_to_absolute_timestamps(dataframes)
        except ValueError as e:
            print(f"❌ {e}")

    print(f"\n📊 Successfully extracted {len(dataframes)} sensor types from {filepath.name}")
    return dataframes


def write_dfs(dfs: dict[str, pd.DataFrame], output_path: Path):
    for sensor_name, df in dfs.items():
        df.to_csv(output_path / f"{sensor_name}.csv", index=False)


def write_single_df(dfs: dict[str, pd.DataFrame], output_path: Path):
    """
    Write a single sparse CSV aligned on the union of all sensor timestamps.
    """
    combined_df = build_sparse_union_frame(dfs)

    if combined_df.empty:
        # Create empty file with header for consistency
        (output_path / "all.csv").write_text("", encoding="utf-8")
        return

    combined_df.to_csv(output_path / "all.csv", index=False)


def build_sparse_union_frame(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a sparse, outer-aligned DataFrame using the union of all timestamps.

    Args:
        dfs: Mapping of sensor name -> DataFrame with "timestamp" column.

    Returns:
        DataFrame with one row per unique timestamp and sensor columns aligned by timestamp.
    """
    if not dfs:
        return pd.DataFrame()

    # Filter valid dataframes (non-empty with timestamp column)
    valid_dfs = []
    for sensor_name, df in dfs.items():
        if df is not None and not df.empty and "timestamp" in df.columns:
            # Handle duplicate timestamps by keeping first occurrence
            clean_df = df.drop_duplicates(subset=["timestamp"], keep="first")
            valid_dfs.append(clean_df)

    if not valid_dfs:
        return pd.DataFrame()

    # Merge all dataframes on timestamp using outer join
    combined_df = reduce(
        lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), valid_dfs
    )

    # Sort by timestamp and reset index
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

    # Ensure timestamp is first column
    cols = ["timestamp"] + [col for col in combined_df.columns if col != "timestamp"]
    combined_df = combined_df[cols]

    return combined_df


# Example usage:
# dfs_dict = {
#     'aalp': aalp_df,
#     'accl': accl_df,
#     'gps9': gps9_df
# }
# write_single_df(dfs_dict, Path('.'))


def make_plots(df):
    pass


@click.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--config-path", type=click.Path(exists=True), default="sensors.json")
@click.option("--multiple", is_flag=True, default=False)
@click.option("--single", is_flag=True, default=False)
@click.option("--output-path", type=click.Path(), default=".")
@click.option(
    "--absolute",
    is_flag=True,
    default=False,
    help="Convert timestamps to absolute Unix epoch nanoseconds using GPS reference",
)
def main(filepath, config_path, multiple, single, output_path, absolute):
    dfs = make_dfs(Path(filepath), Path(config_path), absolute_time=absolute)

    # Create output directory based on video file location and name
    video_path = Path(filepath)
    if output_path == ".":
        # Default: create folder with video name in same directory as video
        output_path = video_path.parent / video_path.stem
    else:
        output_path = Path(output_path)
        # If specified path doesn't exist, create it
        if not output_path.is_dir():
            output_path = output_path / video_path.stem

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")

    if multiple:
        write_dfs(dfs, output_path)
        print(f"✅ Wrote separate CSV files to {output_path}")
    if single:
        write_single_df(dfs, output_path)
        print(f"✅ Wrote combined CSV file to {output_path}")

    if not multiple and not single:
        print("Info: Use --multiple or --single flags to write CSV files")

    print("\n📊 DataFrames summary:")
    for name, df in dfs.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
