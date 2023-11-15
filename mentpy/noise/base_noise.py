# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
from abc import ABCMeta, abstractmethod


class BaseNoise(metaclass=ABCMeta):
    """BaseNoise clase"""

    def __init__(self):
        """Initialize the current object"""
