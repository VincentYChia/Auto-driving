"""Shared, read-only access to NAVSIM data, used by both workstreams.

* ``navsim_eval/`` (scoring models on NAVSIM) and ``data_pipeline/`` (training data) both
  build on this package; neither imports the other.
* ``geometry``, ``raw_frames`` and ``maneuvers`` (what the driver did over the next 4 s) are pure numpy and
  import without NAVSIM installed.
* ``loader`` needs the NAVSIM devkit, so it runs in the NAVSIM conda env (Python 3.9).
"""
