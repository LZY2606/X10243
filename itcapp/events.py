"""Event layer: every state change is a machine-readable event.

Events carry: stable id, project, branch, monotonic seq, type, actor,
payload (JSON) and the material versions visible at that moment
(material_refs). Replaying a branch up to event N restores exactly the
dependency versions and visible conclusions of that moment.
"""
from __future__ import annotations

# event types
PROJECT_CREATED = "project_created"
MATERIAL_IMPORTED = "material_imported"          # payload: slot, version_id, content_hash, diagnostics
MATERIAL_SUPERSEDED = "material_superseded"      # payload: slot, old_version, new_version
INTEGRATION_COMPUTED = "integration_computed"    # payload: result_id, input_summary, param_version
INTEGRATION_ENDPOINT_ADJUSTED = "integration_endpoint_adjusted"
FIRST_INJECTION_TOGGLED = "first_injection_toggled"
BLANK_RULE_SET = "blank_rule_set"                # payload: rule name + rationale (machine readable)
BLANK_SUBTRACTION_COMPUTED = "blank_subtraction_computed"
MODEL_FIT_COMPUTED = "model_fit_computed"        # payload: candidates incl. ties
MODEL_ACCEPTED = "model_accepted"
MODEL_REJECTED = "model_rejected"
MODEL_PARAMS_ADJUSTED = "model_params_adjusted"
BRANCH_CREATED = "branch_created"
STALE_MARKED = "stale_marked"                    # payload: which derived results went stale and why

# judgment codes used inside payloads (machine readable)
JUDG_PEAK_OVERLAP_SPLIT = "peak_overlap_split"
JUDG_GAP_CLIPPED = "integration_window_clipped_at_gap"
JUDG_FIRST_INJECTION_DISCARDED = "first_injection_discarded"
JUDG_BLANK_MISSING = "blank_correspondence_missing"
JUDG_BLANK_RULE = "dilution_rule"
JUDG_MODEL_STALE_BY_CONCENTRATION = "model_stale_by_concentration_revision"
JUDG_MODEL_TIE = "multiple_equally_good_models"
JUDG_ENDPOINT_MANUAL = "integration_endpoint_manual"

ALL_TYPES = {
    PROJECT_CREATED, MATERIAL_IMPORTED, MATERIAL_SUPERSEDED,
    INTEGRATION_COMPUTED, INTEGRATION_ENDPOINT_ADJUSTED,
    FIRST_INJECTION_TOGGLED, BLANK_RULE_SET, BLANK_SUBTRACTION_COMPUTED,
    MODEL_FIT_COMPUTED, MODEL_ACCEPTED, MODEL_REJECTED,
    MODEL_PARAMS_ADJUSTED, BRANCH_CREATED, STALE_MARKED,
}


def make_event(event_id, project_id, branch, seq, etype, payload, material_refs, actor="reviewer"):
    if etype not in ALL_TYPES:
        raise ValueError(f"unknown event type {etype}")
    return {
        "id": event_id,
        "project_id": project_id,
        "branch": branch,
        "seq": seq,
        "type": etype,
        "actor": actor,
        "payload": payload,
        "material_refs": material_refs,
    }
