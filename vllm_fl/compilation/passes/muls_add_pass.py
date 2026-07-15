# Copyright (c) 2026 BAAI. All rights reserved.

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import init_logger

from vllm_fl.compilation.passes.base_pattern import BasePattern

logger = init_logger(__name__)


class MulsAddPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, scale: float = 1.0):
        super().__init__(vllm_config)
        self.scale = scale

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(2, 2048, device="npu", dtype=self.dtype)
        y = torch.randn(2, 2048, device="npu", dtype=self.dtype)
        return [x, y]

    def get_pattern(self):
        def pattern(x: torch.Tensor, y: torch.Tensor):
            return x * self.scale + y

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, y: torch.Tensor):
            return torch.ops.vllm.muls_add(x, y, self.scale)

        return replacement


class MulsAddFusionPass(VllmInductorPass):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(
            pass_name="muls_add_fusion_pass"
        )

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            logger.debug("MulsAdd fusion not enabled: unsupported dtype %s", dtype)
            return

        routed_scaling_factor = getattr(
            vllm_config.model_config.hf_text_config, "routed_scaling_factor", 1.0
        )
        MulsAddPattern(vllm_config, scale=routed_scaling_factor).register(
            self.pattern_match_passes
        )

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s muls_add patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
