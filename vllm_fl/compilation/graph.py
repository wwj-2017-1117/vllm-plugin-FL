# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/compilation/cuda_graph.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import weakref
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_graph_wrappers: weakref.WeakSet[Any] = weakref.WeakSet()
_STREAM_RESOURCE_ERROR_CODE = "207008"
_STREAM_RESOURCE_ERROR_MARKERS = (
    "insufficient_stream_resources",
    "stream resources are insufficient",
)
_STREAM_RESOURCE_GUIDANCE = (
    "NPU graph capture failed with a known stream-resource exhaustion "
    "signature. Consider upgrading to a newer HDK/CANN stack, reducing "
    "cudagraph_capture_sizes, lowering max_cudagraph_capture_size, preferring "
    "FULL or FULL_DECODE_ONLY for mostly uniform decode workloads, or "
    "temporarily disabling graph mode to confirm the failure is capture-related."
)


def _is_stream_resource_capture_error(exc: RuntimeError) -> bool:
    message = str(exc)
    lowered_message = message.lower()
    has_error_code = _STREAM_RESOURCE_ERROR_CODE in message
    has_stream_resource_marker = any(
        marker in lowered_message for marker in _STREAM_RESOURCE_ERROR_MARKERS
    )
    return has_stream_resource_marker or (
        has_error_code and "stream resource" in lowered_message
    )


def _raise_stream_resource_capture_error(exc: RuntimeError) -> None:
    raise RuntimeError(f"{_STREAM_RESOURCE_GUIDANCE}\nOriginal error:\n{exc}") from exc


def _is_draft_eagle_graph(use_eagle: bool) -> bool:
    if not use_eagle:
        return False
    try:
        from vllm_fl.ascend_forward_context import _EXTRA_CTX
    except ImportError:
        return False
    return bool(getattr(_EXTRA_CTX, "is_draft_model", False))


def weak_ref_tensors(tensor: Any) -> Any:
    if current_platform.device_type == "cuda":
        from vllm.utils.torch_utils import weak_ref_tensors
        return weak_ref_tensors(tensor)
    else:
        ### TODO: add csrc npu custom op
        return tensor


class Graph:
    if current_platform.device_type == "cuda":
        graph = torch.cuda.CUDAGraph
    elif current_platform.device_type == "npu":
        graph = torch.npu.NPUGraph
    elif current_platform.device_type == "musa":
        graph = torch.musa.MUSAGraph
    elif current_platform.device_type == "txda":
        graph = None
    else:
        raise NotImplementedError("not support graph")

@dataclasses.dataclass
class GraphEntry:
    batch_descriptor: BatchDescriptor
    graph: Optional[Graph] = None
    output: Optional[Any] = None

    # for graph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None

@dataclasses.dataclass
class GraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


class GraphWrapper:
    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: Optional[GraphOptions] = None,
        *,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = GraphOptions()
        self.graph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_graph_entries: dict[BatchDescriptor, GraphEntry] = {}
        self.enable_enpu = enable_enpu
        self.use_eagle = use_eagle
        _graph_wrappers.add(self)

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of graph wrapper: "
                f"{self._runnable_str}"
            )
        raise AttributeError(
            f"Attribute {key} not found. Set VLLM_LOGGING_LEVEL=DEBUG for "
            "more details."
        )

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        graph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            graph_runtime_mode == CUDAGraphMode.NONE
            or graph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_graph_entries:
            # create a new entry for this batch descriptor
            self.concrete_graph_entries[batch_descriptor] = GraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_graph_entries[batch_descriptor]

        if entry.graph is None:
            if self.graph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            graph = Graph.graph()

            output = None
            with ExitStack() as stack:
                if self.graph_options.gc_disable:
                    # during every model forward for piecewise graph
                    # mode, we will capture many pieces of graphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the graph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("vllm_fl.platform.PlatformFL.empty_cache", lambda: None)
                    )

                set_graph_pool_id(self.graph_pool)

                is_npu_graph = current_platform.device_type == "npu"
                if is_npu_graph:
                    forward_context.capturing = True
                try:
                    # mind-exploding: carefully manage the reference and memory.
                    with current_platform.torch_device_fn.graph(
                        graph, pool=self.graph_pool
                    ):
                        # `output` is managed by pytorch's graph pool
                        output = self.runnable(*args, **kwargs)
                        if self.graph_options.weak_ref_output:
                            # by converting it to weak ref,
                            # the original `output` will immediately be released
                            # to save memory. It is only safe to do this for
                            # the last graph in piecewise graph mode, because
                            # the output of the last graph will not be used by
                            # any other graph.
                            output = weak_ref_tensors(output)
                except RuntimeError as exc:
                    if is_npu_graph and _is_stream_resource_capture_error(exc):
                        _raise_stream_resource_capture_error(exc)
                    raise
                finally:
                    if is_npu_graph:
                        forward_context.capturing = False

            if current_platform.device_type == "npu":
                weak_ref_workspaces(_graph_params)
                weak_ref_workspaces(_draft_graph_params)
                weak_ref_workspaces(_draft_graph_prefill_params)

            entry.output = weak_ref_tensors(output)
            entry.graph = graph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        if current_platform.device_type == "npu":
            is_full_graph = self.runtime_mode == CUDAGraphMode.FULL
            need_sync = is_full_graph and not _is_draft_eagle_graph(self.use_eagle)
            if not self.enable_enpu and need_sync:
                torch.npu.current_stream().synchronize()
        else:
            current_platform.torch_device_fn.synchronize()
        entry.graph.replay()
        return entry.output


def weak_ref_workspaces(params):
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(params.workspaces[num_tokens])


def update_full_graph_params(
    attn_backend,
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    speculative_config=None,
    num_dcp_pcp_tokens=None,
    draft_attn_metadatas=None,
):
    impl_cls = attn_backend.get_impl_cls()
    impl_cls.update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config,
        num_dcp_pcp_tokens,
        draft_attn_metadatas,
    )


@dataclass
class GraphParams:
    events: dict[int, list[Any]]
    workspaces: dict[int, torch.Tensor | None]
    handles: dict[int, list[Any]]
    attn_params: dict[int, list[tuple]]


_graph_params: GraphParams | None = None


def set_graph_params(graph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: [] for size in graph_capture_sizes},
        {size: None for size in graph_capture_sizes},
        {size: [] for size in graph_capture_sizes},
        {size: [] for size in graph_capture_sizes},
    )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = workspace


def get_graph_params():
    return _graph_params


_draft_graph_params: GraphParams | None = None


def set_draft_graph_params(graph_capture_sizes: list[int]):
    global _draft_graph_params
    if _draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    _draft_graph_params = GraphParams(
        {size: [] for size in graph_capture_sizes},
        {size: None for size in graph_capture_sizes},
        {size: [] for size in graph_capture_sizes},
        {size: [] for size in graph_capture_sizes},
    )


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_params
    if _draft_graph_params is not None:
        _draft_graph_params.workspaces[num_tokens] = workspace


def get_draft_graph_params():
    return _draft_graph_params


_draft_graph_prefill_params: GraphParams | None = None


def set_draft_graph_prefill_params(graph_capture_sizes: list[int]):
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph prefill parameters have already been set!")
    _draft_graph_prefill_params = GraphParams(
        {size: [] for size in graph_capture_sizes},
        {size: None for size in graph_capture_sizes},
        {size: [] for size in graph_capture_sizes},
        {size: [] for size in graph_capture_sizes},
    )


def update_draft_graph_prefill_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        _draft_graph_prefill_params.workspaces[num_tokens] = workspace


def get_draft_graph_prefill_params():
    return _draft_graph_prefill_params


ACLGraphEntry = GraphEntry
ACLGraphWrapper = GraphWrapper
ACLGraphOptions = GraphOptions
set_acl_graph_params = set_graph_params
update_acl_graph_params_workspaces = update_graph_params_workspaces
get_acl_graph_params = get_graph_params
