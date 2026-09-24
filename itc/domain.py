"""Domain materials: immutable, separately versioned.

Kinds: power_trace, injection_schedule, concentrations, temperature,
cell_volume, baseline, blank_curve. Each gets its own version sequence
and a stable identity  mat:<kind>:v<n>  that survives export/import.
"""
import hashlib
import json

KINDS = ("power_trace", "injection_schedule", "concentrations", "temperature",
         "cell_volume", "baseline", "blank_curve")


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def digest_of(payload):
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:16]


def material_id(kind, version):
    return "mat:%s:v%d" % (kind, version)


def make_material(kind, version, payload):
    return {
        "id": material_id(kind, version),
        "kind": kind,
        "version": version,
        "payload": payload,
        "digest": digest_of(payload),
    }


def input_digest(materials, params):
    """Digest over the exact inputs of a computation: material digests plus
    the parameter version, so results are traceable to their parents."""
    parts = [(m["id"], m["digest"]) for m in materials]
    parts.sort()
    return digest_of({"inputs": parts, "params": params})
