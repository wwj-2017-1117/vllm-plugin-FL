# Copyright (c) 2026 BAAI. All rights reserved.

from torch._inductor.pattern_matcher import Match
from vllm.logger import init_logger

logger = init_logger(__name__)


def extra_stream_scope_check(match: Match) -> bool:
    """
    Checks if all nodes in the same stream.
    """
    non_default_streams = set()
    has_default = False

    for node in match.nodes:
        if node.op == "call_function":
            current_stream = node.meta.get("stream_label")
            if current_stream is None:
                has_default = True
            else:
                non_default_streams.add(current_stream)
                if len(non_default_streams) > 1:
                    logger.debug(
                        "Cross-stream operation detected in pattern match "
                        "for AddRMSNormQuant. Multiple streams found: %s. "
                        "Fusion is not supported for cross-stream operations.",
                        non_default_streams,
                    )
                    return False

    if has_default and len(non_default_streams) > 0:
        logger.debug(
            "Cross-stream operation detected in pattern match for "
            "AddRMSNormQuant. Multiple streams found: %s. Fusion is not "
            "supported for cross-stream operations.",
            non_default_streams,
        )
        return False

    return True


_register_patterns = set()


def check_and_register_fusion_pass(pattern_class: type, **kwargs):
    global _register_patterns
    eps = kwargs.get("eps", 1e-6)
    pattern_key = str(pattern_class.__name__) + str(eps)
    if pattern_key in _register_patterns:
        return

    pattern = pattern_class(**kwargs)
    try:
        pattern.register()
        _register_patterns.add(pattern_key)
    except RuntimeError as e:
        if "Duplicate pattern" in str(e):
            logger.warning(
                "Pattern %s eps %s has been registered",
                pattern_class.__name__,
                eps,
            )
            _register_patterns.add(pattern_key)
        else:
            raise e
