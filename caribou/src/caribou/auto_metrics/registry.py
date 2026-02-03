from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import sys
from typing import Dict, List, Optional


AUTO_METRICS_DIR = Path(__file__).resolve().parent
_DISCOVERED = False
_REGISTRY: Dict[str, "MetricEntry"] = {}


@dataclass(frozen=True)
class MetricSpec:
    id: str
    name: str
    description: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    tags: List[str]
    version: str = "v1"
    default: bool = False


@dataclass(frozen=True)
class MetricEntry:
    spec: MetricSpec
    cls: type
    module_path: Path
    class_name: str
    init_kwargs: Dict[str, object]


def register_metric(
    spec: MetricSpec,
    cls: type,
    *,
    init_kwargs: Optional[Dict[str, object]] = None,
) -> None:
    if spec.id in _REGISTRY:
        raise ValueError(f"Duplicate metric id: {spec.id}")
    module_path = Path(inspect.getfile(cls)).resolve()
    _REGISTRY[spec.id] = MetricEntry(
        spec=spec,
        cls=cls,
        module_path=module_path,
        class_name=cls.__name__,
        init_kwargs=init_kwargs or {},
    )


def discover_metrics() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return

    if str(AUTO_METRICS_DIR) not in sys.path:
        sys.path.insert(0, str(AUTO_METRICS_DIR))

    for path in AUTO_METRICS_DIR.glob("*.py"):
        if path.name in {"AutoMetric.py", "registry.py", "__init__.py"}:
            continue
        module_name = f"caribou.auto_metrics.{path.stem}"
        if module_name in sys.modules:
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            print(f"[caribou] Skipping metric module {path.name}: {exc}")

    _DISCOVERED = True


def list_metrics(tag: Optional[str] = None) -> List[MetricSpec]:
    discover_metrics()
    specs = [entry.spec for entry in _REGISTRY.values()]
    if tag:
        specs = [spec for spec in specs if tag in spec.tags]
    return sorted(specs, key=lambda s: s.id)


def get_metric_entry(metric_id: str) -> MetricEntry:
    discover_metrics()
    return _REGISTRY[metric_id]


def find_metric_id_by_path(path: Path) -> Optional[str]:
    discover_metrics()
    resolved = path.resolve()
    for metric_id, entry in _REGISTRY.items():
        if entry.module_path == resolved:
            return metric_id
    return None
