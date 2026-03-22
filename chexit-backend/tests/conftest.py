"""Avoid hitting Google Drive when tests import ``app.main`` (lifespan runs on TestClient)."""

from __future__ import annotations

import os

os.environ.setdefault("CHEXIT_SKIP_GDOWN", "1")
