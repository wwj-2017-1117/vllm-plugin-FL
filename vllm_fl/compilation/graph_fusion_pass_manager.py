# Copyright (c) 2026 BAAI. All rights reserved.

from torch import fx as fx
from vllm.compilation.passes.inductor_pass import get_pass_context
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class GraphFusionPassManager:
    """Pass manager for Ascend graph fusion passes."""

    def __init__(self):
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph) -> fx.Graph:
        compile_range = get_pass_context().compile_range
        total_matches = 0

        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
                total_matches += getattr(pass_, "matched_count", 0)
        graph.recompile()
        if total_matches:
            logger.info(
                "Applied FL graph fusion passes for compile_range=%s; "
                "total_replacements=%d",
                compile_range,
                total_matches,
            )
        return graph

    def add(self, pass_: VllmInductorPass):
        assert isinstance(pass_, VllmInductorPass)
        self.passes.append(pass_)

    def configure(self, config: VllmConfig):
        device = getattr(getattr(config, "device_config", None), "device", None)
        if device is not None and str(device).split(":", 1)[0] != "npu":
            return

        additional_config = config.additional_config or {}
        ascend_compilation_config = additional_config.get(
            "ascend_compilation_config", {}
        )

        if ascend_compilation_config.get("fuse_norm_quant", True):
            from vllm_fl.compilation.passes.norm_quant_fusion_pass import (
                AddRMSNormQuantFusionPass,
            )

            self.passes.append(AddRMSNormQuantFusionPass(config))

        if ascend_compilation_config.get("fuse_qknorm_rope", True):
            from vllm_fl.compilation.passes.qknorm_rope_fusion_pass import (
                QKNormRopeFusionPass,
            )

            self.passes.append(QKNormRopeFusionPass(config))

        if ascend_compilation_config.get("fuse_allreduce_rms", True):
            from vllm_fl.compilation.passes.allreduce_rmsnorm_fusion_pass import (
                MatmulAllReduceAddRMSNormPass,
            )

            self.passes.append(MatmulAllReduceAddRMSNormPass(config))

        if ascend_compilation_config.get("fuse_muls_add", True):
            from vllm_fl.compilation.passes.muls_add_pass import MulsAddFusionPass

            self.passes.append(MulsAddFusionPass(config))

        if self.passes:
            logger.info_once(
                "Enabled FL graph fusion passes: %s",
                [pass_.__class__.__name__ for pass_ in self.passes],
            )
