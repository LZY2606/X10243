"""Orchestration: materials -> derived computations -> events.

Every derived result records its input digest, parameter version and
parent dependencies. Domain judgements are emitted as machine-readable
events, e.g.:

- integration.overlap_clamped   post-injection peaks may overlap; the
                                window ends at the next injection
- integration.gap_blocked       an integration window must not cross a
                                data gap
- dilution.rule_declared        dilution-heat subtraction always states
                                its blank-correspondence rule
- model.staled                  a concentration revision expires models
                                but never rewrites the original integrals
"""

import hashlib
import json

from . import integrate as integ
from . import models as mdl
from . import units
from .errors import (OK, SEMANTIC_UNSOLVABLE if False else OK, diag)  # placeholder
