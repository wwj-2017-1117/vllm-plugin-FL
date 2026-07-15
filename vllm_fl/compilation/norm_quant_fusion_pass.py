# Copyright (c) 2026 BAAI. All rights reserved.

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import init_logger

from vllm_fl.compilation.passes.base_pattern import BasePattern

logger = init_logger(__name__)


class AddRMSNormDynamicQuantPattern(BasePattern):
    def get_inputs(self):
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
        ):
            output = torch.ops.npu.npu_add_rms_norm(
                rms_norm_input, residual, rms_norm_weight, self.eps
            )
            quantized_output = torch.ops.npu.npu_dynamic_quant(output[0])
            return quantized_output[0], quantized_output[1], output[2]

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
        ):
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                epsilon=self.eps,
                output_mask=[True, True],
            )
            return output[0], output[3], output[2]

        return replacement


class AddRMSNormDynamicQuantSPPattern(AddRMSNormDynamicQuantPattern):
    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
        ):
            output = torch.ops.npu.npu_add_rms_norm(
                rms_norm_input, residual, rms_norm_weight, self.eps
            )
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(output[0], True)
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], output[2]

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
        ):
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                epsilon=self.eps,
                output_mask=[True, True],
            )
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                output[0], True
            )
            scale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(output[3], True)
            return quantized_output, scale, output[2]

        return replacement


class AddRMSNormQuantFusionPass(VllmInductorPass):
    """Fuse AddRMSNorm + dynamic quant into npu_add_rms_norm_dynamic_quant."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(
            pass_name="rmsnorm_quant_fusion_pass"
        )

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("Quant fusion not enabled: unsupported dtype %s", dtype)
            return

        for eps in [1e-5, 1e-6]:
            AddRMSNormDynamicQuantPattern(vllm_config, eps=eps).register(
                self.pattern_match_passes
            )
            AddRMSNormDynamicQuantSPPattern(vllm_config, eps=eps).register(
                self.pattern_match_passes
            )

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Replaced %s AddRMSNormQuant patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
