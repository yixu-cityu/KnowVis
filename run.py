import argparse
import json
from pathlib import Path
from time import perf_counter


def load_config(config_path: Path) -> tuple[Path | None, str]:
    config_path = config_path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("target_video"), dict):
        raise ValueError("Config must contain a target_video object")
    target = config["target_video"]
    local_path = target.get("local_path", "")
    youtube_url = target.get("youtube_url", "")
    if any(value is not None and not isinstance(value, str) for value in (local_path, youtube_url)):
        raise ValueError("local_path and youtube_url must be strings or null")
    local_path = (local_path or "").strip()
    youtube_url = (youtube_url or "").strip()
    if not local_path and not youtube_url:
        raise ValueError("Set target_video.local_path or target_video.youtube_url")
    video_dir = Path(local_path).expanduser() if local_path else None
    if video_dir is not None and not video_dir.is_absolute():
        video_dir = (config_path.parent / video_dir).resolve()
    return video_dir, youtube_url


def main():
    parser = argparse.ArgumentParser(description="Process a local or YouTube video")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.json"),
        help="Path to config.json; defaults to the file beside run.py",
    )
    args = parser.parse_args()
    try:
        video_dir, youtube_url = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    from video_processor.parser import process_video

    start = perf_counter()
    print("Start video processing", flush=True)
    chunks = process_video(video_dir=video_dir, youtube_url=youtube_url)
    print(f"Finished: {len(chunks)} chunks in {perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
