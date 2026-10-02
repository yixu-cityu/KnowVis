import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve user input once, relative to its explicit owning directory."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = (base_dir if base_dir is not None else Path.cwd()) / path
    return path.resolve()


def read_gemini_api_key(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    """Read the credential exclusively from the selected JSON configuration."""
    config_path = resolve_path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    api_key = config.get('GEMINI_API_KEY') if isinstance(config, dict) else None
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f'GEMINI_API_KEY must be a non-empty string in {config_path}')
    return api_key.strip()
