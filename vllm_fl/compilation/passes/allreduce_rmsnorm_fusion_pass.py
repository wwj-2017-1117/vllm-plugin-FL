# Copyright (c) 2026 BAAI. All rights reserved.

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from vllm.compilation.passes.inductor_pass import get_pass_context
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

from vllm_fl.compilation.passes.base_pattern import BasePattern
from vllm_fl.compilation.passes.utils.npugraph_ex_utils_check import (
    extra_stream_scope_check,
)

logger = init_logger(__name__)
ALLREDUCE_NORM_FUSE_THRESHOLD = 512


def get_compile_range_and_extra_stream_check():
    def check_func(match) -> bool:
        compile_range = get_pass_context().compile_range
        return (
            extra_stream_scope_check(match)
            and compile_range.start > ALLREDUCE_NORM_FUSE_THRESHOLD
        )

    return check_func


class MatmulAllReduceAddRMSNormPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)
        device_group = get_tp_group().device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.local_rank = torch.distributed.get_rank(group=device_group)
        self.tp_group_name = backend.get_hccl_comm_name(self.local_rank)
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        batch_size, seq_len, hidden_size = 2, 4, 4096
        x = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        weight = torch.randn(hidden_size, hidden_size, device="npu")
        residual = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        rms_norm_weight = torch.randn(hidden_size, device="npu")
        return [x, weight, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(x, weight, residual, rms_norm_weight):
            mm = torch.ops.vllm.unquantized_gemm(x, weight, None)
            all_reduce = tensor_model_parallel_all_reduce(mm)
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                all_reduce, residual, rms_norm_weight, None
            )
            return output[0], output[2]

        return pattern

    def get_replacement(self):
        def replacement(x, weight, residual, rms_norm_weight):
            return torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
                x,
                weight,
                residual,
                rms_norm_weight,
                self.tp_group_name,
                self.tp_size,
                self.local_rank,
                self.eps,
                True,
                False,
            )

        return replacement

    def get_extra_stream_scope_check(self):
        return get_compile_range_and_extra_stream_check()


class MatmulAllReduceAddRMSNormPass(VllmInductorPass):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(
            pass_name="allreduce_rmsnorm_fusion_pass"
        )

        if vllm_config.parallel_config.tensor_parallel_size <= 1:
            logger.debug("AllReduce RMSNorm fusion skipped when tensor_parallel_size <= 1")
            return

        MatmulAllReduceAddRMSNormPattern(vllm_config).register(
            self.pattern_match_passes
        )

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        pattern_idx = 0
        for pattern_entry in self.pattern_match_passes.patterns.values():
            for pattern in pattern_entry:
                pattern_str = PatternPrettyPrinter.run(pattern.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, pattern_str)
                pattern_idx += 1
        if self.matched_count:
            logger.info(
                "Optimized %s patterns with allreduce_rmsnorm_fusion_pass: "
                "matmul + all_reduce + add_rms_norm -> "
                "matmul_allreduce_add_rmsnorm",
                self.matched_count,
            )
        else:
            logger.debug("Replaced %s allreduce rmsnorm patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.start > ALLREDUCE_NORM_FUSE_THRESHOLD
