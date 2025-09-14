import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import click
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path to import the telemetry extractor
sys.path.append(str(Path(__file__).parent.parent))
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor


def extract_gopro_face_data(video_path: Path) -> pd.DataFrame:
    """
    Extract FACE data directly from GoPro MP4 file.
    
    Args:
        video_path: Path to GoPro MP4 file
        
    Returns:
        DataFrame with columns: timestamp, face_ver, face_confidence, face_id, 
        face_x, face_y, face_w, face_h, face_smile, face_blink
    """
    extractor = GoProTelemetryExtractor(video_path)
    extractor.open_source()
    
    try:
        face_data, face_timestamps = extractor.extract_data("FACE")
        
        if len(face_data) == 0:
            print("⚠️  No FACE data found in video file")
            return pd.DataFrame()
        
        # GoPro FACE structure: ver, confidence%, ID, x, y, w, h, smile%, blink%
        face_df = pd.DataFrame(face_data, columns=[
            'face_ver', 'face_confidence', 'face_id', 'face_x', 'face_y', 
            'face_w', 'face_h', 'face_smile', 'face_blink'
        ])
        face_df['timestamp'] = face_timestamps
        
        print(f"✅ Extracted {len(face_df)} FACE data points from video")
        return face_df
        
    finally:
        extractor.close_source()


def draw_face_boxes(
    video_path: Path,
    csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    frame_rate: Optional[float] = None,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    box_thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.6,
) -> None:
    """
    Draw GoPro FACE bounding boxes on video frames.
    
    Args:
        video_path: Path to input GoPro MP4 video file
        csv_path: Path to CSV file with face data (if None, extracts from video)
        output_path: Path for output video (if None, uses input name + '_faces')
        frame_rate: Output video frame rate (uses input rate if None)
        box_color: Default RGB color for bounding boxes (B, G, R for OpenCV)
        box_thickness: Thickness of bounding box lines
        text_color: RGB color for text labels
        font_scale: Scale factor for text font size
        
    Note:
        GoPro FACE data structure: version, confidence%, ID, x, y, w, h, smile%, blink%
        Coordinates are normalized (0-1) and automatically converted to pixels.
        Box colors change based on confidence: Green (70%+), Yellow (40-70%), Red (<40%)
    """
    # Load or extract face detection data
    if csv_path:
        try:
            face_df = pd.read_csv(csv_path)
            print(f"📁 Loaded face data from CSV: {len(face_df)} rows")
            print(f"Columns: {list(face_df.columns)}")
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
    else:
        print("🎥 Extracting FACE data directly from video...")
        face_df = extract_gopro_face_data(video_path)
        if face_df.empty:
            print("❌ No face data available. Exiting.")
            return
    
    # Set default output path if not provided
    if output_path is None:
        output_path = video_path.with_stem(f"{video_path.stem}_faces")
    
    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fps = frame_rate if frame_rate else input_fps
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Input FPS: {input_fps:.2f}")
    print(f"  Output FPS: {output_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp for this frame
            timestamp = frame_idx / input_fps
            
            # Find face data for this timestamp (within tolerance)
            tolerance = 1.0 / input_fps  # One frame duration tolerance
            frame_faces = face_df[
                (face_df['timestamp'] >= timestamp - tolerance) & 
                (face_df['timestamp'] <= timestamp + tolerance)
            ]
            
            # Draw bounding boxes for detected faces
            for _, face_row in frame_faces.iterrows():
                # Extract GoPro FACE data structure: ver, confidence%, ID, x, y, w, h, smile%, blink%
                try:
                    # GoPro FACE data columns (adjust based on your CSV export)
                    version = face_row.get('face_ver', face_row.get('face_version', 4))
                    confidence = face_row.get('face_confidence', 0)  # Already in percentage
                    face_id = int(face_row.get('face_id', 0))
                    x = face_row.get('face_x', 0)
                    y = face_row.get('face_y', 0)
                    w = face_row.get('face_w', 0)
                    h = face_row.get('face_h', 0)
                    smile = face_row.get('face_smile', 0)  # Already in percentage
                    blink = face_row.get('face_blink', 0)  # Already in percentage
                    
                    # GoPro coordinates are typically normalized (0-1 range)
                    # Convert to pixel coordinates
                    if x <= 1.0 and y <= 1.0 and w <= 1.0 and h <= 1.0:
                        x_px = int(x * width)
                        y_px = int(y * height)
                        w_px = int(w * width)
                        h_px = int(h * height)
                    else:
                        # Already in pixel coordinates
                        x_px, y_px, w_px, h_px = int(x), int(y), int(w), int(h)
                    
                    # Skip if bounding box is invalid
                    if w_px <= 0 or h_px <= 0:
                        continue
                    
                    # Choose box color based on confidence (green = high, yellow = medium, red = low)
                    if confidence >= 70:
                        current_box_color = (0, 255, 0)  # Green
                    elif confidence >= 40:
                        current_box_color = (0, 255, 255)  # Yellow
                    else:
                        current_box_color = (0, 0, 255)  # Red
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x_px, y_px), (x_px + w_px, y_px + h_px), 
                                current_box_color, box_thickness)
                    
                    # Prepare label text with GoPro FACE data
                    label_parts = [f"ID:{face_id}"]
                    if confidence > 0:
                        label_parts.append(f"Conf:{confidence:.0f}%")
                    if smile > 0:
                        label_parts.append(f"😊{smile:.0f}%")
                    if blink > 0:
                        label_parts.append(f"😑{blink:.0f}%")
                    
                    label = " ".join(label_parts)
                    
                    # Calculate text size and position
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                    )
                    
                    # Draw text background
                    cv2.rectangle(
                        frame,
                        (x, y - text_height - baseline - 5),
                        (x + text_width, y),
                        box_color,
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame,
                        label,
                        (x, y - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        1,
                        cv2.LINE_AA
                    )
                    
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid face data at frame {frame_idx}: {e}")
                    continue
            
            # Write frame to output video
            out.write(frame)
            
            # Progress indicator
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Processing: {progress:.1f}% ({frame_idx}/{total_frames})")
            
            frame_idx += 1
    
    finally:
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"✅ Video with bounding boxes saved to: {output_path}")


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False)
@click.option("--csv-path", type=click.Path(exists=True), help="CSV file with face data (if not provided, extracts from video)")
@click.option("--fps", type=float, help="Output video frame rate (uses input rate if not specified)")
@click.option("--color", default="0,255,0", help="Default bounding box color as B,G,R values (confidence overrides this)")
@click.option("--thickness", default=2, help="Bounding box line thickness")
@click.option("--font-scale", default=0.6, help="Text font scale factor")
def main(video_path, output_path, csv_path, fps, color, thickness, font_scale):
    """
    Draw GoPro FACE bounding boxes on video using telemetry data.
    
    Extracts FACE data directly from GoPro MP4 files or uses provided CSV.
    
    Examples:
        # Extract FACE data from video and draw boxes
        python draw_bounds.py video.mp4
        
        # Use existing CSV data
        python draw_bounds.py video.mp4 output.mp4 --csv-path face_data.csv
        
        # Custom styling
        python draw_bounds.py video.mp4 output.mp4 --fps 30 --thickness 3
    """
    # Parse color
    try:
        box_color = tuple(map(int, color.split(',')))
        if len(box_color) != 3:
            raise ValueError("Color must have 3 values")
    except ValueError:
        raise click.BadParameter("Color must be in format 'B,G,R' (e.g., '0,255,0')")
    
    draw_face_boxes(
        video_path=Path(video_path),
        csv_path=Path(csv_path) if csv_path else None,
        output_path=Path(output_path) if output_path else None,
        frame_rate=fps,
        box_color=box_color,
        box_thickness=thickness,
        font_scale=font_scale,
    )


if __name__ == "__main__":
    main()
