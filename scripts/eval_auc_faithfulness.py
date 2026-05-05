#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for unified AUC faithfulness evaluation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attribution_research.evaluation.auc_faithfulness import main


if __name__ == "__main__":
    main()
