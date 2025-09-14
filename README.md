# py-gpmf-parser

Python bindings for the GoPro Metadata Format (GPMF) parser, allowing for easy extraction of telemetry data from GoPro videos. 

## Overview

This repository offers a Pythonic interface to the GPMF parser, enabling users to extract sensor data like ACCL, GYRO, GPS5, GRAV, and more from GoPro videos.

It provides access to most of the low-level functions (have a look at the **src/gpmf_bindings.cpp**), but also includes a simple python class to extract the telemetry data. Extending it to more low-level stuff should be easy, just follow the structure that I used.

## Requirements

- Python 3.12 or higher
- UV (recommended) or pip for package management
- C++ compiler (for building the native extensions)

### Dependencies

This package includes comprehensive data science and visualization tools:
- **numpy 2.0+**: Core numerical computing
- **pandas 2.0+**: Data manipulation and analysis
- **matplotlib 3.7+**: Static plotting and visualization
- **plotly 5.15+**: Interactive plotting and dashboards
- **jupyter**: Notebook environment for analysis
- **nominal**: Advanced data analysis and visualization

## Installation

### Using UV (Recommended)

This project uses modern Python packaging with UV for dependency management:

```bash
# Clone the repository
git clone --recursive https://github.com/urbste/py-gpmf-parser/
cd py-gpmf-parser

# Install the package and dependencies
uv sync

# Build and install in development mode
uv run python -m pip install -e .
```


### Data Analysis and Visualization

With the included data science libraries, you can easily analyze and visualize telemetry data:

```python
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor

# Extract data
filepath = 'gpmf-parser/samples/hero8.mp4'
extractor = GoProTelemetryExtractor(filepath)
extractor.open_source()

# Get accelerometer and GPS data
accl, accl_t = extractor.extract_data("ACCL")
gps, gps_t = extractor.extract_data("GPS5")
extractor.close_source()

# Create pandas DataFrames for analysis
accl_df = pd.DataFrame(accl, columns=['x', 'y', 'z'])
accl_df['timestamp'] = accl_t

gps_df = pd.DataFrame(gps, columns=['lat', 'lon', 'alt', 'speed_2d', 'speed_3d'])
gps_df['timestamp'] = gps_t

# Static plotting with matplotlib
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(accl_df['timestamp'], accl_df[['x', 'y', 'z']])
plt.title('Accelerometer Data')
plt.legend(['X', 'Y', 'Z'])

plt.subplot(2, 1, 2)
plt.plot(gps_df['timestamp'], gps_df['speed_2d'])
plt.title('GPS Speed')
plt.show()

# Interactive plotting with plotly
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=accl_df['x'], y=accl_df['y'], z=accl_df['z'], 
                          mode='markers', name='Acceleration'))
fig.update_layout(title='3D Acceleration Visualization')
fig.show()
```

## Development

### Setting up Development Environment

This project uses UV for dependency management and modern Python tooling:

```bash
# Clone with submodules
git clone --recursive https://github.com/urbste/py-gpmf-parser/
cd py-gpmf-parser

# Pin Python version (ensures consistency across team)
uv python pin 3.12

# Install development dependencies
uv sync

# Install the package in editable mode
uv run python -m pip install -e .

# Run tests
uv run python -m pytest tests/

# Run linting and type checking
uv run ruff check .
uv run ruff format .
uv run mypy py_gpmf_parser/

# Start Jupyter Lab for interactive analysis
uv run jupyter lab
```

### Building Distribution Packages

```bash
# Build wheel and source distribution
uv build

# The built packages will be in dist/
ls dist/
```

### Code Quality Tools

This project uses:
- **Ruff**: For linting and code formatting
- **mypy**: For static type checking  
- **pytest**: For testing
- **JupyterLab**: For interactive data analysis and notebook development

All tools are configured in `pyproject.toml` and can be run via UV.
