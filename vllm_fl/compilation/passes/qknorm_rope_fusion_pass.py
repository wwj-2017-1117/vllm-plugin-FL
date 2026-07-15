# Copyright (c) 2026 BAAI. All rights reserved.

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention

from vllm_fl.compilation.passes.base_pattern import BasePattern

logger = init_logger(__name__)


def get_rope_dim(vllm_config: VllmConfig) -> int:
    model_config = vllm_config.model_config
    if model_config.use_mla:
        return model_config.hf_text_config.qk_rope_head_dim

    rope_dim = model_config.get_head_size()
    if hasattr(model_config.hf_text_config, "partial_rotary_factor"):
        rope_dim = int(rope_dim * model_config.hf_text_config.partial_rotary_factor)
    elif hasattr(model_config.hf_text_config, "rotary_dim"):
        rope_dim = int(model_config.hf_text_config.rotary_dim)
    return rope_dim


class QKNormRopeFusionPattern(BasePattern):
    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads, eps=1e-6):
        super().__init__(vllm_config, eps)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rope_dim = get_rope_dim(vllm_config)

    def get_inputs(self):
        num_tokens = 5
        max_position_embeddings = 16384
        qkv = torch.empty(
            num_tokens,
            self.q_size + 2 * self.kv_size,
            dtype=torch.bfloat16,
            device="npu",
        )
        q_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        cos_sin_cache = torch.empty(
            max_position_embeddings, self.head_dim, dtype=torch.bfloat16, device="npu"
        )
        positions = torch.ones(num_tokens, dtype=torch.int64, device="npu")
        return [qkv, q_weight, k_weight, cos_sin_cache, positions]

    def get_pattern(self):
        def pattern(qkv, q_weight, k_weight, cos_sin_cache, positions):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(
                q_by_head, q_weight, self.eps
            )
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(
                k_by_head, k_weight, self.eps
            )
            q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
                positions,
                q_norm_out.view(q.shape),
                k_norm_out.view(k.shape),
                cos_sin_cache,
                self.head_dim,
                self.rope_dim,
                True,
            )
            return q_rope, k_rope, v

        return pattern

    def get_replacement(self):
        def replacement(qkv, q_weight, k_weight, cos_sin_cache, positions):
            return torch.ops.vllm.qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.eps,
                q_bias=None,
                k_bias=None,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
            )

        return replacement


class QKNormRopeFusionPass(VllmInductorPass):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(
            pass_name="qknorm_rope_fusion_pass"
        )

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16,):
            logger.debug("QKNorm and Rope fusion not enabled: unsupported dtype %s", dtype)
            return

        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            vllm_config, Attention
        )
        if len(attn_layers) == 0:
            logger.debug(
                "QKNorm and Rope fusion enabled, but no Attention layers were discovered."
            )
            return

        layer = next(iter(attn_layers.values()))
        for epsilon in [1e-6, 1e-5]:
            if layer.head_size != 128:
                logger.debug(
                    "QKNorm and Rope fusion not enabled: head_dim %d is not 128",
                    layer.head_size,
                )
                continue
            QKNormRopeFusionPattern(
                vllm_config=vllm_config,
                head_dim=layer.head_size,
                num_heads=layer.num_heads,
                num_kv_heads=layer.num_kv_heads,
                eps=epsilon,
            ).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s QKNorm and Rope patterns", self.matched_count)
        pattern_idx = 0
        for pattern_entry in self.pattern_match_passes.patterns.values():
            for pattern in pattern_entry:
                pattern_str = PatternPrettyPrinter.run(pattern.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, pattern_str)
                pattern_idx += 1
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
