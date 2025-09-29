"""
NexusFusion Inference Package
============================

This package provides high-level APIs and utilities for running inference
with the NexusFusion multi-modal fusion architecture.

Authors: NexusFusion Research Team  
License: MIT
"""

from .nexus_fusion_api import NexusFusionAPI, create_api, quick_predict

__all__ = [
    'NexusFusionAPI',
    'create_api', 
    'quick_predict'
]
