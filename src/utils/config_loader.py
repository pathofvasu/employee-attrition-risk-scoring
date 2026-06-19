from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _merge_dicts(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def _parse_scalar(value):
    value = value.strip().strip('"').strip("'")
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_load(text):
    root = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = raw_line.strip().partition(":")

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if raw_value.strip():
            parent[key] = _parse_scalar(raw_value)
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))

    return root


def _load_yaml(path):
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _simple_yaml_load(text)


def load_config():
    config = {}
    for filename in ("config.yaml", "model_config.yaml", "paths.yaml", "thresholds.yaml"):
        path = CONFIG_DIR / filename
        if path.exists():
            _merge_dicts(config, _load_yaml(path))
    return config


def project_path(relative_path):
    return PROJECT_ROOT / relative_path
