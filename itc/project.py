"""Workflow service: immutable materials + derived computations + decisions.

Layering:
  * materials are immutable and separately versioned per kind;
  * computations record input digests, parameter versions and parents;
  * decisions are events; conflicting decisions are all retained;
  * checking out an old event restores that moment's dependency versions
    and visible conclusions.
"""

from .. import __name__ as _pkg  # noqa: F401  (keeps package import explicit)
