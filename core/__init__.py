"""wateraudit: Industrial Wastewater Treatment Plant (WWTP) Compliance & Soft-Sensing."""
from core.schema import WWTPDatasetLoader, UCI_ATTRIBUTES, TARGET_COLUMNS
from core.softsense import QuantileSoftSensor, ConformalInterval
from core.compliance import RegulatoryComplianceGuard, ComplianceStatus
from core.efficiency import SettlerEfficiencyAnalyzer, PlantDiagnosticReport
from core.visualizer import WaterAuditVisualizer

__all__ = [
    "WWTPDatasetLoader",
    "UCI_ATTRIBUTES",
    "TARGET_COLUMNS",
    "QuantileSoftSensor",
    "ConformalInterval",
    "RegulatoryComplianceGuard",
    "ComplianceStatus",
    "SettlerEfficiencyAnalyzer",
    "PlantDiagnosticReport",
    "WaterAuditVisualizer",
]
