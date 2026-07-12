"""Minimal YAML config with attribute access and dotted overrides."""

import copy

import yaml


class Cfg(dict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides=None):
    """overrides: list of 'a.b.c=value' strings (YAML-parsed values).
    A top-level `_base: <relative path>` key inherits + deep-merges."""
    import os
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("_base"))
        with open(base_path) as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return Cfg(cfg)
