from functools import cache
from html import unescape
import json
from pathlib import Path
import re


@cache
def load_whisper_model():
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("large-v3", device=device)


def transcribe_video(video_path: Path) -> Path:
    video_path = Path(video_path)
    transcript_path = video_path.with_name(f"transcribed_{video_path.stem}.srt")
    if not transcript_path.is_file():
        from whisper.utils import get_writer

        model = load_whisper_model()
        result = model.transcribe(
            str(video_path), fp16=model.device.type == "cuda", verbose=True
        )
        writer = get_writer("srt", str(video_path.parent))
        writer(result, str(transcript_path))
    return transcript_path


def time_to_seconds(time_str):
    seconds = 0.0
    for part in time_str.replace(",", ".").split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def srtTranscript2json(srt_file):
    transcript_path = Path(srt_file)
    content = transcript_path.read_text(encoding="utf-8-sig")
    timestamp = r"((?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3})"
    result = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.splitlines()
        for index, line in enumerate(lines):
            match = re.match(rf"{timestamp}\s+-->\s+{timestamp}", line)
            if not match:
                continue
            start, end = (round(time_to_seconds(value), 3) for value in match.groups())
            text = " ".join(lines[index + 1:]).strip()
            if transcript_path.suffix.lower() == ".vtt":
                text = unescape(re.sub(r"<[^>]*>", "", text))
            if text:
                if result and result[-1]["text"] == text:
                    result[-1]["end"] = end
                else:
                    result.append({"start": start, "end": end, "text": text})
            break
    return {str(index): segment for index, segment in enumerate(result)}


def get_video_duration(video_path: Path) -> float:
    metadata_path = video_path.with_name(f"{video_path.stem}_metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        duration = metadata.get("video_duration")
        if duration is not None and float(duration) > 0:
            return float(duration)

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    finally:
        capture.release()
    if duration <= 0:
        raise ValueError(f"Unable to determine video duration: {video_path}")
    return duration


def chunk(video_path: Path, slides_timestamps: dict) -> dict:
    video_path = Path(video_path)
    video_length = get_video_duration(video_path)
    chunk_ts = sorted({
        0.0, video_length,
        *(float(slide["start_ts"]) for slide in slides_timestamps.values()
          if 0 < float(slide["start_ts"]) < video_length),
    })
    threshold = max(40, min(video_length * 0.08, 180))
    filtered_ts = [0.0]
    for timestamp in chunk_ts[1:]:
        if timestamp - filtered_ts[-1] >= threshold:
            filtered_ts.append(timestamp)
        elif timestamp == video_length:
            if len(filtered_ts) == 1:
                filtered_ts.append(timestamp)
            else:
                filtered_ts[-1] = timestamp

    final_chunk_ts = [0.0]
    for start, end in zip(filtered_ts, filtered_ts[1:]):
        if end - start > threshold * 2.5:
            final_chunk_ts.append((start + end) / 2)
        final_chunk_ts.append(end)

    transcript_files = sorted(
        path for path in video_path.parent.iterdir()
        if path.is_file() and path.suffix.lower() in {".srt", ".vtt"}
    )
    transcript_files.sort(key=lambda path: (
        not path.name.endswith(("en.srt", "en-US.srt", "en.vtt", "en-US.vtt")),
        path.suffix.lower() != ".srt",
    ))
    transcript_path = transcript_files[0] if transcript_files else transcribe_video(video_path)
    transcript = srtTranscript2json(transcript_path)

    chunks = {}
    visited = set()
    for index, (start, end) in enumerate(zip(final_chunk_ts, final_chunk_ts[1:])):
        texts = []
        for speech_id, segment in transcript.items():
            if speech_id not in visited and float(segment["start"]) < end:
                texts.append(segment["text"])
                visited.add(speech_id)
        chunks[str(index)] = {
            "transcript_text": " ".join(texts),
            "slides": [
                slide_id for slide_id, slide in slides_timestamps.items()
                if start <= float(slide["start_ts"]) < end
            ],
        }

    for index in range(len(chunks) - 1):
        current_chunk = chunks[str(index)]
        next_chunk = chunks[str(index + 1)]
        next_text = next_chunk["transcript_text"]
        dot_index = next_text.find(".")
        if dot_index != -1:
            current_chunk["transcript_text"] = (
                current_chunk["transcript_text"] + " " + next_text[:dot_index + 1]
            ).strip()
            next_chunk["transcript_text"] = next_text[dot_index + 1:].lstrip()
    return chunks
