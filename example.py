from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
filepath = 'GX010021.mp4'
extractor = GoProTelemetryExtractor(filepath)
extractor.open_source()
accl, accl_t = extractor.extract_data("ACCL")
print(accl)
gps, gps_t = extractor.extract_data("GPS5")
extractor.close_source()