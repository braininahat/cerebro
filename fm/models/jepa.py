"""Wrapper around Braindecode's SignalJEPA."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from braindecode.models import SignalJEPA

from cerebro.constants import N_CHANS, SFREQ


def _default_chs_info(n_chans: int) -> list[dict]:
    chs_info = []
    for idx in range(n_chans):
        angle = 2 * math.pi * idx / max(n_chans, 1)
        loc = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=float)
        chs_info.append({"loc": loc})
    return chs_info


def _sanitize_chs_info(chs_info: Sequence[dict], n_chans: int) -> list[dict]:
    if not chs_info:
        return _default_chs_info(n_chans)
    sanitized = []
    replacements = 0
    for idx, ch in enumerate(chs_info):
        loc = ch.get("loc") if isinstance(ch, dict) else None
        if loc is None or len(loc) < 3 or np.isnan(loc[:3]).any() or np.allclose(loc[:3], 0):
            replacements += 1
            angle = 2 * math.pi * idx / max(n_chans, 1)
            loc = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=float)
            sanitized.append({"loc": loc})
        else:
            sanitized.append({"loc": np.array(loc, dtype=float)})
    if replacements == len(sanitized):
        return _default_chs_info(n_chans)
    return sanitized


class EEGJEPAModel(SignalJEPA):
    """SignalJEPA with sensible defaults for EEG2025 data."""

    def __init__(
        self,
        n_chans: int = N_CHANS,
        n_times: int = int(2.0 * SFREQ),
        chs_info: Sequence[dict] | None = None,
        sfreq: float = SFREQ,
        **kwargs,
    ) -> None:
        if chs_info is None:
            chs_info = _default_chs_info(n_chans)
        chs_info = _sanitize_chs_info(chs_info, n_chans)
        super().__init__(
            n_chans=n_chans,
            chs_info=list(chs_info),
            n_times=n_times,
            sfreq=sfreq,
            **kwargs,
        )


__all__ = ["EEGJEPAModel"]
