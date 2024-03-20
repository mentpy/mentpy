# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the flow module."""

import pytest

from mentpy.mbqc import templates
from mentpy.mbqc import flow


def test_cflow():
    gs = templates.linear_cluster(5).graph
    cond, _, _, _ = flow.find_cflow(gs, set([0]), set([4]))
    assert cond


def test_gflow():
    gs = templates.linear_cluster(5).graph
    cond, _, _, _ = flow.find_gflow(gs, set([0]), set([4]))
    assert cond


def test_Pflow():
    gs = templates.linear_cluster(5).graph
    cond, p, d = flow.find_pflow(gs, set([0]), set([4]), {v: "XY" for v in gs.nodes})
    assert cond
