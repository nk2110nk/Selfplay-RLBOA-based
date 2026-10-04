from sao.my_negotiators import reset
from negmas.sao import SAONegotiator, AspirationNegotiator

SAONegotiator.reset = reset
if not hasattr(SAONegotiator, "on_ufun_changed"):
    SAONegotiator.on_ufun_changed = lambda self: None
if not hasattr(AspirationNegotiator, "on_ufun_changed"):
    AspirationNegotiator.on_ufun_changed = lambda self: None

# The old NegMAS release needed local XML/utility compatibility patches.  Newer
# releases removed ``negmas.helpers.ikeys`` and already implement these paths.
try:  # pragma: no cover - exercised by the legacy Docker image
    from sao.my_utilities import from_xml_str, luaf_call, muf_call
    from negmas.utilities import (
        UtilityFunction, LinearUtilityAggregationFunction, MappingUtilityFunction,
    )
    UtilityFunction.from_xml_str = from_xml_str
    LinearUtilityAggregationFunction.__call__ = luaf_call
    MappingUtilityFunction.__call__ = muf_call
except ImportError:
    pass
