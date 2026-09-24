"""Unit tests for settler stage decomposition and fault diagnostics."""
import numpy as np
import pytest
from core.efficiency import SettlerEfficiencyAnalyzer


def test_log_ratio_efficiency():
    analyzer = SettlerEfficiencyAnalyzer()
    # 200 mg/L in, 20 mg/L out -> 90% removal
    l_ratio = analyzer.log_ratio_efficiency(200.0, 20.0)
    assert np.isclose(l_ratio, np.log(0.10))


def test_hydraulic_shock_diagnosis():
    analyzer = SettlerEfficiencyAnalyzer()
    # Flow surge: 58000 m3/day (normal ~37000), deteriorated removals
    report = analyzer.evaluate_stages(
        q_in=58000.0,
        ss_in=250.0,
        ss_primary=180.0,
        ss_out=80.0,
        bod_in=180.0,
        bod_out=60.0,
        cod_in=400.0,
        cod_out=180.0,
        sed_out=1.8,
    )
    top_hypothesis = report.fault_hypotheses[0][0]
    assert top_hypothesis == "Hydraulic Shock Load"


def test_normal_operation_diagnosis():
    analyzer = SettlerEfficiencyAnalyzer()
    # Normal flow, excellent removals (>85% removal)
    report = analyzer.evaluate_stages(
        q_in=36000.0,
        ss_in=250.0,
        ss_primary=100.0,
        ss_out=15.0,
        bod_in=200.0,
        bod_out=12.0,
        cod_in=400.0,
        cod_out=45.0,
        sed_out=0.1,
    )
    assert report.global_bod_removal > 90.0
    assert report.global_ss_removal > 90.0
