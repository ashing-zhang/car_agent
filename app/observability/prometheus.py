# Observability Prometheus 导出 - 文本展示格式(规格第20节,Phase 8)
# 运行指南:
#   from app.observability.prometheus import render_prometheus
#   text = render_prometheus()  # 供 GET /metrics 使用

from __future__ import annotations

from typing import Any


def _escape_label(value: str) -> str:
    """转义 label 值中的特殊字符。"""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: dict[str, str]) -> str:
    """格式化 label 集合为 Prometheus 文本。"""
    if not labels:
        return ""
    parts = [f'{k}="{_escape_label(v)}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"


def render_prometheus(metrics_registry: object) -> str:
    """将 MetricsRegistry 渲染为 Prometheus exposition 文本。"""
    lines: list[str] = []
    counters = metrics_registry.counters()  # type: ignore[attr-defined]
    histograms = metrics_registry.histograms()  # type: ignore[attr-defined]

    for name, counter in counters.items():
        lines.append(f"# HELP {name} {counter.help_text}")
        lines.append(f"# TYPE {name} counter")
        for labels, value in counter.samples():
            lines.append(f"{name}{_format_labels(labels)} {value}")

    for name, hist in histograms.items():
        lines.append(f"# HELP {name} {hist.help_text}")
        lines.append(f"# TYPE {name} histogram")
        for labels, counts, total_sum, total in hist.samples():
            cumulative = 0
            for i, bound in enumerate(hist.buckets):
                cumulative = counts[i]
                lbl = dict(labels)
                lbl["le"] = str(bound)
                lines.append(f"{name}_bucket{_format_labels(lbl)} {cumulative}")
            inf_labels = dict(labels)
            inf_labels["le"] = "+Inf"
            lines.append(f"{name}_bucket{_format_labels(inf_labels)} {counts[-1]}")
            lines.append(f"{name}_sum{_format_labels(labels)} {total_sum}")
            lines.append(f"{name}_count{_format_labels(labels)} {total}")

    return "\n".join(lines) + "\n"


def render_snapshot(snapshot: dict[str, Any]) -> str:
    """可选:从 snapshot dict 渲染(便于测试)。"""
    lines: list[str] = []
    for name, samples in snapshot.get("counters", {}).items():
        lines.append(f"# TYPE {name} counter")
        for s in samples:
            lines.append(f"{name}{_format_labels(s['labels'])} {s['value']}")
    for name, samples in snapshot.get("histograms", {}).items():
        lines.append(f"# TYPE {name} histogram")
        for s in samples:
            lines.append(f"{name}_count{_format_labels(s['labels'])} {s['total']}")
            lines.append(f"{name}_sum{_format_labels(s['labels'])} {s['sum']}")
    return "\n".join(lines) + "\n"
