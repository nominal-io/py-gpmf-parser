import pandas as pd
import numpy as np
from pathlib import Path
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
import click
import nominal
import json


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
    filepath: Path, config_path: Path) -> dict[str, pd.DataFrame]:
    """
    Extract telemetry data from GoPro MP4 file and return as separate DataFrames.

    Args:
        filepath: Path to the MP4 file.
        config_path: Filesystem path to the sensors JSON configuration that
            defines available FOURCC keys, axis names, and descriptions.

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

    print(f"\n📊 Successfully extracted {len(dataframes)} sensor types from {filepath.name}")
    return dataframes


def write_dfs(dfs: dict[str, pd.DataFrame], output_path: Path):
    for sensor_name, df in dfs.items():
        df.to_csv(output_path / f"{sensor_name}.csv", index=False)

def write_single_df(dfs: dict[str, pd.DataFrame], output_path: Path):
    # concat the dfs into a sparse dataframe
    df = pd.concat(dfs.values(), axis=1)
    df.to_csv(output_path / "all.csv", index=False)

def make_plots(df):
    pass


@click.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--config-path", type=click.Path(exists=True), default="sensors.json")
@click.option("--multiple", is_flag=True, default=False)
@click.option("--single", is_flag=True, default=False)
@click.option("--output-path", type=click.Path(), default=".")
def main(filepath, config_path, multiple, single, output_path):
    dfs = make_dfs(Path(filepath), Path(config_path))

    # check if the output path is a directory and if not make it based on vid file name
    if not Path(output_path).is_dir():
        output_path = Path(output_path) / Path(filepath).stem
        output_path.mkdir(parents=True, exist_ok=True)

    if multiple:
        write_dfs(dfs, Path(output_path))
    if single:
        write_single_df(dfs, Path(output_path))
    print(dfs)


if __name__ == "__main__":
    main()
