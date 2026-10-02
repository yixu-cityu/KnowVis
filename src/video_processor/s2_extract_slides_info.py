import json
from pathlib import Path

import cv2
import imagehash
from PIL import Image


def _save_slide(path: Path, frame) -> None:
    # OpenCV's Windows filename handling can reject Unicode paths. Encoding the
    # same PNG in memory lets pathlib perform the Unicode-aware file operation.
    success, encoded = cv2.imencode('.png', frame)
    if not success:
        raise OSError(f"Unable to encode slide: {path}")
    path.write_bytes(encoded.tobytes())


def remove_duplicates(base_dir, raw_timestamps, hash_size=12, threshold=5):
    base_dir = Path(base_dir)
    timestamps = {name: value.copy() for name, value in raw_timestamps.items()}
    kept_file = None
    kept_hash = None
    for filename in sorted(raw_timestamps, key=lambda name: (len(name), name)):
        try:
            with Image.open(base_dir / filename) as image:
                current_hash = imagehash.dhash(image, hash_size=hash_size)
        except (OSError, ValueError):
            continue
        if kept_hash is not None and kept_hash - current_hash <= threshold:
            timestamps[kept_file]["end_ts"] = max(
                timestamps[kept_file]["end_ts"], timestamps[filename]["end_ts"]
            )
            (base_dir / filename).unlink()
            del timestamps[filename]
        else:
            kept_file = filename
            kept_hash = current_hash
    return timestamps


def capture_slides_frame_diff(
    video_path, output_dir_path, MIN_PERCENT_THRESH=0.06, ELAPSED_FRAME_THRESH=85
):
    output_dir = Path(output_dir_path)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        success, first_frame = capture.read()
        if not success:
            raise ValueError(f"Unable to read video frames: {video_path}")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        previous_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        screenshot_count = 1
        filename = f"temp_{screenshot_count:03}.png"
        _save_slide(output_dir / filename, first_frame)
        timestamps = {}
        frame_index = 1
        start_frame = 0
        pending_start = None
        elapsed_frames = 0

        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(current_frame, previous_frame)
            _, frame_diff = cv2.threshold(frame_diff, 160, 255, cv2.THRESH_BINARY)
            frame_diff = cv2.dilate(frame_diff, kernel)
            changed_percent = cv2.countNonZero(frame_diff) / current_frame.size * 100

            if changed_percent >= MIN_PERCENT_THRESH and pending_start is None:
                pending_start = frame_index
            elif pending_start is not None:
                elapsed_frames += 1

            if pending_start is not None and elapsed_frames >= ELAPSED_FRAME_THRESH:
                timestamps[filename] = (start_frame, pending_start - 1)
                start_frame = pending_start
                screenshot_count += 1
                filename = f"temp_{screenshot_count:03}.png"
                _save_slide(output_dir / filename, frame)
                pending_start = None
                elapsed_frames = 0

            previous_frame = current_frame
            frame_index += 1

        timestamps[filename] = (start_frame, frame_index - 1)
        return {
            name: {"start_ts": round(start / fps, 3), "end_ts": round(end / fps, 3)}
            for name, (start, end) in timestamps.items()
        }
    finally:
        capture.release()


def rename_files_sequentially(base_dir, filenames=None):
    base_dir = Path(base_dir)
    if filenames is None:
        filenames = (path.name for path in base_dir.glob("*.png"))
    rename_map = {}
    for index, old_name in enumerate(sorted(filenames, key=lambda name: (len(name), name)), 1):
        new_name = f"{index:03}.png"
        (base_dir / old_name).replace(base_dir / new_name)
        rename_map[old_name] = new_name
    return rename_map


def extract_slides(video_file_path: Path, output_dir_path: Path) -> dict:
    output_dir = Path(output_dir_path)
    timestamps_path = output_dir / "slides_timestamps.json"
    if timestamps_path.is_file():
        with timestamps_path.open("r", encoding="utf-8") as file:
            cached_timestamps = json.load(file)
        if cached_timestamps and all((output_dir / name).is_file() for name in cached_timestamps):
            return cached_timestamps

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_timestamps = capture_slides_frame_diff(video_file_path, output_dir)
    timestamps = remove_duplicates(output_dir, raw_timestamps)
    rename_map = rename_files_sequentially(output_dir, timestamps)
    result = {new_name: timestamps[old_name] for old_name, new_name in rename_map.items()}
    with timestamps_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)
    return result
