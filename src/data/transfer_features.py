"""Single source of truth for the 3 features shared between Beijing (R2,
the transfer source) and Fontibon (the transfer target), used for R2 and
the eventual A/B/C/D transfer experiments.

Both domains' prepared feature frames MUST expose these exact names, in
this exact order — channel i must mean the same physical concept in both
Beijing R2 and Fontibon, or a transferred encoder's input projection
would silently mix up variables between the source and target domains.

Determined by auditing the actual code (not assumed):
- src/data/fontibon.py: load_fontibon_features returns
  X[["pm25", "temp", "dew"]] — established since Primera Entrega, not
  changed here.
- src/data/beijing.py: R1's own CONTINUOUS_COLS uses "dewp" (not "dew")
  and orders dew point BEFORE temperature — a different name AND a
  different order from Fontibon's. R2 does NOT reuse R1's column list;
  it builds its own 3-column frame directly in this canonical order (see
  src/data/beijing.py: prepare_beijing_r2).

See docs/IMPLEMENTATION_ASSUMPTIONS.md, "R2 — transfer source
pretraining", for the full audit trail.
"""
TRANSFER_FEATURE_NAMES = ["pm25", "temp", "dew"]
