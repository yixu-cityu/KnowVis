import json
from pathlib import Path

from .s1_prepare_raw_data import prepare_raw_data
from ..common.paths import DATA_DIR, resolve_path


class VideoProcessor:
    def __init__(self, video_dir: str | Path | None = None, youtube_url: str = ""):
        self.video_dir = resolve_path(video_dir) if video_dir is not None else None
        self.youtube_url = youtube_url
        self.data_dir = DATA_DIR
        self._video_path: Path | None = None

    @property
    def video_path(self) -> Path:
        if self._video_path is None:
            raise RuntimeError("Call process() before accessing the resolved video path")
        return self._video_path

    @property
    def output_dir(self) -> Path:
        return resolve_path(self.data_dir) / "processed" / self.video_path.stem

    def process(self) -> dict:
        video_path = prepare_raw_data(self.video_dir, self.youtube_url)
        self._video_path = video_path
        video_id = video_path.stem
        output_dir = self.output_dir
        output_path = output_dir / f"transcribe_chunk_{video_id}.json"
        if output_path.is_file():
            cached_chunks = json.loads(output_path.read_text(encoding="utf-8"))
            slide_names = {name for item in cached_chunks.values() for name in item['slides']}
            if slide_names and all((output_dir / 'slides' / name).is_file() for name in slide_names):
                return cached_chunks

        from .s2_extract_slides_info import extract_slides
        from .s3_segmentation import chunk

        slides = extract_slides(video_path, output_dir / "slides")
        chunks = chunk(video_path, slides)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, ensure_ascii=False, indent=4)
        return chunks


def process_video(video_dir: str | Path | None = None, youtube_url: str = "") -> dict:
    return VideoProcessor(video_dir, youtube_url).process()
