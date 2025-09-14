import py_gpmf_parser as pgfp
import numpy as np
import os
from pathlib import Path
from typing import Union

class GoProTelemetryExtractor:
    def __init__(self, mp4_filepath: Union[str, Path]):
        """
        Initialize the GoPro telemetry extractor.
        
        Args:
            mp4_filepath: Path to the MP4 file (string or Path object)
        
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the file is not a valid MP4 file
        """
        # Convert Path objects to string and validate
        if isinstance(mp4_filepath, Path):
            self.mp4_filepath = str(mp4_filepath.resolve())
        else:
            self.mp4_filepath = str(mp4_filepath)
        
        # Check if file exists
        if not os.path.exists(self.mp4_filepath):
            raise FileNotFoundError(f"MP4 file not found: {self.mp4_filepath}")
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(self.mp4_filepath):
            raise ValueError(f"Path is not a file: {self.mp4_filepath}")
        
        # Basic file extension check
        if not self.mp4_filepath.lower().endswith(('.mp4', '.mov')):
            raise ValueError(f"File must be an MP4 or MOV file: {self.mp4_filepath}")
        
        self.handle = None

    def open_source(self):
        """
        Open the MP4 source for telemetry extraction.
        
        Returns:
            int: Handle to the opened source
            
        Raises:
            ValueError: If source is already opened or if opening fails
            RuntimeError: If the MP4 file doesn't contain GPMF data
        """
        if self.handle is None:
            try:
                self.handle = pgfp.OpenMP4Source(
                    self.mp4_filepath, 
                    pgfp.MOV_GPMF_TRAK_TYPE, 
                    pgfp.MOV_GPMF_TRAK_SUBTYPE, 
                    0
                )
                
                # Check if we got a valid handle
                if self.handle <= 0:
                    self.handle = None
                    raise RuntimeError(f"Failed to open MP4 source. The file may not contain GPMF telemetry data: {self.mp4_filepath}")
                
                return self.handle
                
            except Exception as e:
                self.handle = None
                if "incompatible function arguments" in str(e):
                    raise ValueError(f"Invalid file path format: {self.mp4_filepath}") from e
                else:
                    raise RuntimeError(f"Failed to open MP4 source: {e}") from e
        else:
            raise ValueError("Source is already opened!")

    def close_source(self):
        if self.handle is not None:
            pgfp.CloseSource(self.handle)
            self.handle = None
        else:
            raise ValueError("No source to close!")

    def get_image_timestamps_s(self):
        if self.handle is None:
            raise ValueError("Source is not opened!")

        num_frames, numer, denom = pgfp.GetVideoFrameRateAndCount(self.handle)
        frametime = denom / numer
        timestamps = []
        for i in range(num_frames):
            timestamps.append(i*frametime)
        return np.array(timestamps)

    def extract_data(self, sensor_type: str):
        """
        Extract telemetry data for a specific sensor type.
        
        Args:
            sensor_type: The sensor type to extract (e.g., "ACCL", "GYRO", "GPS5")
            
        Returns:
            tuple: (data_array, timestamps_array) - numpy arrays of sensor data and timestamps
            
        Raises:
            ValueError: If source is not opened or sensor type is invalid
            RuntimeError: If extraction fails
        """
        if self.handle is None:
            raise ValueError("Source is not opened! Call open_source() first.")

        if not isinstance(sensor_type, str) or len(sensor_type) != 4:
            raise ValueError(f"Sensor type must be a 4-character string, got: {sensor_type}")

        results = []
        timestamps = []

        try:
            rate, start, end = pgfp.GetGPMFSampleRate(self.handle, pgfp.Str2FourCC(sensor_type), pgfp.Str2FourCC("SHUT"))
        except Exception as e:
            raise RuntimeError(f"Failed to get sample rate for sensor '{sensor_type}': {e}") from e

        num_payloads = pgfp.GetNumberPayloads(self.handle)
        for i in range(num_payloads):
            payloadsize = pgfp.GetPayloadSize(self.handle, i)
            res_handle = 0
            res_handle = pgfp.GetPayloadResource(self.handle, res_handle, payloadsize)
            payload = pgfp.GetPayload(self.handle, res_handle, i, payloadsize)
            
            ret, t_in, t_out = pgfp.GetPayloadTime(self.handle, i)

            delta_t = t_out - t_in

            ret, stream = pgfp.GPMF_Init(payload, payloadsize)
            if ret != pgfp.GPMF_ERROR.GPMF_OK:
                continue

            while pgfp.GPMF_ERROR.GPMF_OK == pgfp.GPMF_FindNext(stream, pgfp.Str2FourCC("STRM"), pgfp.GPMF_RECURSE_LEVELS_AND_TOLERANT):
                if pgfp.GPMF_ERROR.GPMF_OK != pgfp.GPMF_FindNext(stream, pgfp.Str2FourCC(sensor_type), pgfp.GPMF_RECURSE_LEVELS_AND_TOLERANT):
                    continue

                key = pgfp.GPMF_Key(stream)
                elements = pgfp.GPMF_ElementsInStruct(stream)
                rawdata = pgfp.GPMF_RawData(stream)
                samples = pgfp.GPMF_Repeat(stream)
                if samples:
                    buffersize = samples * elements * 8
                    ret, data = pgfp.GPMF_ScaledData(stream, buffersize, 0, samples, pgfp.GPMF_SampleType.DOUBLE)
                    data = data[:samples*elements]
                    if pgfp.GPMF_ERROR.GPMF_OK == ret:
                        results.extend(np.reshape(data, (-1, elements)))
                        timestamps.extend([t_in + i*delta_t/samples for i in range(samples)])
            pgfp.GPMF_ResetState(stream)

        # Convert to numpy arrays
        data_array = np.array(results)
        timestamp_array = np.array(timestamps) + start
        
        # Provide helpful feedback if no data was found
        if len(results) == 0:
            print(f"Warning: No '{sensor_type}' data found in the MP4 file. "
                  f"This file may not contain telemetry data for this sensor type.")
        
        return data_array, timestamp_array
    

    def extract_data_to_json(self, json_file, sensor_types=["ACCL", "GYRO"]):
        import json
        out_dict = {"img_timestamps_s": self.get_image_timestamps_s().tolist()}
        for sensor in sensor_types:
            data, timestamps = self.extract_data(sensor)
            out_dict.update({sensor: {"data": data.tolist(), "timestamps_s": timestamps.tolist()}})
        with open(json_file, "w") as f:
            json.dump(out_dict, f)

    def close(self):
        self.close_source()
