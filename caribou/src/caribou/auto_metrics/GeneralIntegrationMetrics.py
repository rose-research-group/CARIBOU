"""
Backward-compatible shim for legacy imports.

Use IntegrationMetric from IntegrationMetrics.py with the "integration_generic" spec.
"""

from caribou.auto_metrics.IntegrationMetrics import IntegrationMetric

__all__ = ["IntegrationMetric"]
