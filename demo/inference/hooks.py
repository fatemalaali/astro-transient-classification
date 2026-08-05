"""Pipeline hook points.

Real/bogus separation is **deliberately out of scope** for this thesis: the
gold set is built from spectroscopically and catalogue-confirmed objects, so
there is no bogus class to learn and no bogus ground truth to evaluate against.
The hook exists so the architecture diagram and the trace panel can show
*where* such a stage would sit, without pretending one is running.
"""

from __future__ import annotations

from demo.models import NormalisedAlert

#: Shown in the trace panel for the skipped stage.
BOGUS_HOOK_NOTE = (
    "Real/bogus filtering is out of scope for this thesis — hook point only. "
    "A bogus stage would sit here, between the adapter and inference, consuming "
    "the cutout triplet and the ZTF rb/drb scores."
)


def bogus_filter(alert: NormalisedAlert) -> bool:
    """HOOK POINT — return True to keep the alert.

    Currently returns True unconditionally. Its single call site in
    ``demo/inference/__init__.py`` is what makes the stage's position in the
    pipeline unambiguous.

    An implementation would consume ``alert.cutouts`` plus ``alert.rb`` /
    ``alert.drb`` — which is the only reason those two ZTF scores are carried on
    the normalised record at all. They are never model features.
    """
    return True
