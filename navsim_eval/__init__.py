"""Workstream A: score models on NAVSIM (currently the MVLAD-AD baseline).

Pipeline: ``export_mvlad_inputs`` (NAVSIM env) -> ``predict_mvlad`` (model env, GPU)
-> ``precomputed_agent`` replayed by NAVSIM's PDM scorer (NAVSIM env).
See ``docs/navsim_eval/RUNBOOK.md``. Independent of ``data_pipeline/``.
"""
