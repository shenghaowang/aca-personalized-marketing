"""Utilities for configuration management and Hydra/OmegaConf custom resolvers."""

from typing import Dict, List

from omegaconf import OmegaConf


def get_collective_feature(collective_list: List[Dict[str, str]]) -> str:
    """Extract feature name from collective criterion list.

    Args:
        collective_list: List of collective criterion dictionaries

    Returns:
        str: The feature name from the first collective criterion

    Raises:
        ValueError: If collective_list is empty or None
    """
    if not collective_list or len(collective_list) == 0:
        raise ValueError("collective_list is empty or None")

    target_feature = collective_list[0]
    keys = list(target_feature.keys())
    return keys[0]


def register_custom_resolvers():
    """Register all custom OmegaConf resolvers for Hydra configurations."""
    if not OmegaConf.has_resolver("get_collective_feature"):
        OmegaConf.register_new_resolver(
            "get_collective_feature",
            get_collective_feature,
        )

    if not OmegaConf.has_resolver("frac_to_pct"):
        OmegaConf.register_new_resolver(
            "frac_to_pct",
            lambda frac_value: str(int(frac_value * 100)),
        )
