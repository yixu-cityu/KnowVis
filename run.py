import argparse
import json
from pathlib import Path
from time import perf_counter

from src.common.paths import DEFAULT_CONFIG_PATH, read_gemini_api_key, resolve_path


def load_config(config_path: str | Path) -> tuple[Path | None, str]:
    config_path = resolve_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
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
    video_dir = resolve_path(local_path, config_path.parent) if local_path else None
    return video_dir, youtube_url


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate KnowVis visual summaries from a local or YouTube video")
    parser.add_argument('--stage', choices=('video', 'all'), default='all',
                        help='Run video preprocessing only, or the full pipeline (default: all)')
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.json; defaults to the file beside run.py",
    )
    args = parser.parse_args(argv)
    try:
        args.config = resolve_path(args.config)
        video_dir, youtube_url = load_config(args.config)
        if args.stage == 'all':
            read_gemini_api_key(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    start = perf_counter()
    print("Start video processing", flush=True)
    try:
        from src.video_processor.parser import VideoProcessor
        processor = VideoProcessor(video_dir=video_dir, youtube_url=youtube_url)
        chunks = processor.process()
        print(f"Video: {processor.video_path}")
        print(f"Output: {processor.output_dir}")
        if args.stage == 'all':
            from src.pipeline import run_knowVis_pipeline
            video_id = processor.video_path.stem
            result = run_knowVis_pipeline(video_id, chunks, processor.output_dir, config_path=args.config)
            print(f"Generated {len(result['visuals'])} visuals")
        print(f"Finished: {len(chunks)} chunks in {perf_counter() - start:.2f}s")
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == "__main__":
    main()
