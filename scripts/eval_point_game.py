#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for Point Game / Energy Point Game evaluation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attribution_research.evaluation.point_game import main


if __name__ == "__main__":
    main()
