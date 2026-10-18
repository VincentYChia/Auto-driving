"""Bridge between this repo and the NAVSIM devkit.

Pure-numpy modules (``geometry``, ``raw_frames``) import without NAVSIM installed.
Modules that need the devkit (``loader``, ``export_mvlad_inputs``, ``precomputed_agent``)
import it lazily, so they must run inside the NAVSIM conda environment (Python 3.9).
"""
