"""Training data v1: the pretraining build in ``docs/training_data/PRETRAIN_V1_PLAN.md``.

Unlike the v0.1 prototype modules in ``data_pipeline/``, this follows the agreed plan: OpenScene
trainval only, native CAM_F0 images, log-level splits from NAVSIM's official lists, gated steps.

* ``safe_pickle``: loads log pickles without running arbitrary code.
* ``navsim_splits``: NAVSIM's split lists, read from a devkit checkout.
* ``tables``: Parquet in and out, as dicts of numpy columns.
* ``openscene_index``: steps 1–2, every valid anchor frame in the metadata (no images).
* ``selection``: rules 3–8, which anchors become samples.
* ``audit``: the Gate 0 report (aggregate statistics only, safe to commit).

Outputs go to ``data/pretrain/<version>/`` in the repo and stay local (git-ignored), except ID lists
and aggregate statistics; see the plan, §5.
"""
