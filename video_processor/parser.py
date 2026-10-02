import json
from pathlib import Path

from .s1_prepare_raw_data import prepare_raw_data
from .s2_extract_slides_info import extract_slides
from .s3_segmentation import chunk


class VideoProcessor:
    def __init__(self, video_dir: Path | None = None, youtube_url: str = ""):
        self.video_dir = video_dir
        self.youtube_url = youtube_url
        self.data_dir = Path(__file__).resolve().parent / "data"

    def process(self) -> dict:
        video_path = prepare_raw_data(self.video_dir, self.youtube_url)
        video_id = video_path.stem
        output_dir = self.data_dir / "processed" / video_id
        slides = extract_slides(video_path, output_dir / "slides")
        chunks = chunk(video_path, slides)
        output_path = output_dir / f"transcribe_chunk_{video_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, ensure_ascii=False, indent=4)
        return chunks


def process_video(video_dir: Path | None = None, youtube_url: str = "") -> dict:
    return VideoProcessor(video_dir, youtube_url).process()
