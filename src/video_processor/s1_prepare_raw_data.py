import json
from pathlib import Path
import re
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import yt_dlp
from ..common.paths import DATA_DIR, PROJECT_ROOT, resolve_path


def get_youtube_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url)
    host = (parsed.hostname or "").lower()
    parts = parsed.path.strip("/").split("/")
    video_id = ""
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parts[0]
    elif host in {
        "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
        "youtube-nocookie.com", "www.youtube-nocookie.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) == 2 and parts[0] in {"embed", "shorts", "live"}:
            video_id = parts[1]
    if parsed.scheme not in {"http", "https"} or not re.fullmatch(r"[\w-]{11}", video_id, re.ASCII):
        raise ValueError(f"Invalid YouTube video URL: {youtube_url}")
    return video_id


def youtube_options() -> dict:
    options = {
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        # YouTube challenge solving requires EJS and a supported JS runtime.
        "js_runtimes": {"deno": {}, "node": {}},
    }
    cookie_file = PROJECT_ROOT / "cookies.txt"
    if cookie_file.is_file():
        options["cookiefile"] = str(cookie_file)
    return options


def youtube_download_transcripts(
    video_id, video_url, output_dir, subtitleslangs=None, subtitlesformat="srt"
):
    options = youtube_options() | {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": subtitleslangs or ["en"],
        "subtitlesformat": subtitlesformat,
        "outtmpl": str(Path(output_dir) / f"transcript_{video_id}.%(ext)s"),
    }
    # yt-dlp accepts an extensible options dict; bundled stubs expose a closed TypedDict.
    with yt_dlp.YoutubeDL(cast(Any, options)) as downloader:
        return downloader.extract_info(video_url, download=True)


def download_video_with_metadata(video_id, video_url, download_dir):
    download_dir = resolve_path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    video_path = download_dir / f"{video_id}.mp4"
    for language in ("en", "en-US"):
        for subtitle_format in ("srt", "vtt"):
            if any(
                path.suffix.lower() in {".srt", ".vtt"}
                for path in download_dir.glob("transcript_*")
            ):
                break
            youtube_download_transcripts(
                video_id, video_url, download_dir, [language], subtitle_format
            )

    options = youtube_options() | {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(video_path),
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(cast(Any, options)) as downloader:
        metadata = downloader.extract_info(video_url, download=not video_path.is_file())
    if not metadata:
        raise RuntimeError(f"Unable to retrieve video metadata: {video_url}")
    return metadata


def prepare_raw_data(video_dir: str | Path | None = None, youtube_url: str = "") -> Path:
    if video_dir:
        video_dir = resolve_path(video_dir)
        if video_dir.is_file():
            if video_dir.suffix.lower() != '.mp4':
                raise ValueError(f"Expected an MP4 video: {video_dir}")
            return video_dir
        if video_dir.exists() and not video_dir.is_dir():
            raise NotADirectoryError(video_dir)
        videos = sorted(
            path for path in video_dir.glob("*")
            if path.is_file() and path.suffix.lower() == ".mp4"
        )
        if len(videos) > 1:
            raise ValueError(f"Expected one MP4 file in {video_dir}, found {len(videos)}")
        if videos:
            return videos[0]
        if not youtube_url:
            raise FileNotFoundError(f"No MP4 video found in {video_dir}")
        if video_dir.suffix.lower() == '.mp4':
            raise FileNotFoundError(f"Video file not found: {video_dir}")

    video_id = get_youtube_video_id(youtube_url)
    if not video_dir:
        video_dir = DATA_DIR / "raw" / video_id
    video_path = video_dir / f"{video_id}.mp4"
    if video_path.is_file():
        return video_path
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    metadata = download_video_with_metadata(video_id, video_url, video_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"Download did not produce the expected MP4: {video_path}")
    video_info = {
        "index": video_id,
        "course_title": "",
        "course_description": "",
        "youtube_id": metadata.get("id", video_id),
        "video_title": metadata.get("title"),
        "video_description": (metadata.get("description") or "").replace("\n", " "),
        "course_channel": metadata.get("uploader"),
        "video_tags": metadata.get("tags"),
        "video_thumbnail": metadata.get("thumbnail"),
        "video_url": metadata.get("webpage_url") or metadata.get("video_url") or video_url,
        "video_width": metadata.get("width"),
        "video_height": metadata.get("height"),
        "video_duration": metadata.get("duration"),
    }
    with (video_dir / f"{video_id}_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(video_info, file, ensure_ascii=False, indent=4)
    return video_path
