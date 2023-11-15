# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.

# TODO: Add AKLT state class inheritance from graph state
import networkx as nx


class ClusterState(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError
