"""
Jittor runtime helpers.
"""

from __future__ import annotations

import logging
import jittor as jt

logger = logging.getLogger(__name__)


def setup_device(use_cuda: bool) -> None:
    if use_cuda and jt.has_cuda:
        jt.flags.use_cuda = 1
        logger.info("Using CUDA")
    else:
        jt.flags.use_cuda = 0
        logger.info("Using CPU")
