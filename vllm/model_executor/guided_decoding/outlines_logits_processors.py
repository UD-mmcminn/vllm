# Copyright 2024- the Outlines developers
# This file is adapted from
# https://github.com/outlines-dev/outlines/blob/main/outlines/serve/vllm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
from collections import defaultdict
from functools import lru_cache
from typing import Callable, DefaultDict, Dict, List, Union

import numpy as np
import torch
from outlines import grammars
from outlines.models import TransformerTokenizer
from outlines.caching import cache
from outlines.fsm.guide import CFGGuide, CFGState, Generate, Guide, RegexGuide, Write
from outlines.fsm.guide_mp import CFGGuideMP
from outlines.fsm.parsing import PartialLark
from outlines.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from line_profiler import LineProfiler
lp = LineProfiler()

import faulthandler
faulthandler.enable()

import traceback

"""
INFO 11-16 17:40:56 engine.py:269] Added request chatcmpl-4397dcc1f3774d6196f406a32d64e51d.
  File "<string>", line 1, in <module>
  File "/usr/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "/usr/lib/python3.12/multiprocessing/spawn.py", line 135, in _main
    return self._bootstrap(parent_sentinel)
  File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/udadmin/LLM/vllm/vllm/engine/multiprocessing/engine.py", line 365, in run_mp_engine
    engine.start()
  File "/home/udadmin/LLM/vllm/vllm/engine/multiprocessing/engine.py", line 133, in start
    self.run_engine_loop()
  File "/home/udadmin/LLM/vllm/vllm/engine/multiprocessing/engine.py", line 196, in run_engine_loop
    request_outputs = self.engine_step()
  File "/home/udadmin/LLM/vllm/vllm/engine/multiprocessing/engine.py", line 205, in engine_step
    return self.engine.step()
  File "/home/udadmin/LLM/vllm/vllm/engine/llm_engine.py", line 1454, in step
    outputs = self.model_executor.execute_model(
  File "/home/udadmin/LLM/vllm/vllm/executor/distributed_gpu_executor.py", line 82, in execute_model
    driver_outputs = self._driver_execute_model(execute_model_req)
  File "/home/udadmin/LLM/vllm/vllm/executor/multiproc_gpu_executor.py", line 158, in _driver_execute_model
    return self.driver_worker.execute_model(execute_model_req)
  File "/home/udadmin/LLM/vllm/vllm/worker/worker_base.py", line 343, in execute_model
    output = self.model_runner.execute_model(
  File "/home/udadmin/LLM/pytorch/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/home/udadmin/LLM/vllm/vllm/worker/model_runner_base.py", line 116, in _wrapper
    return func(*args, **kwargs)
  File "/home/udadmin/LLM/vllm/vllm/worker/model_runner.py", line 1687, in execute_model
    logits = self.model.compute_logits(hidden_or_intermediate_states,
  File "/home/udadmin/LLM/vllm/vllm/model_executor/models/llama.py", line 563, in compute_logits
    logits = self.logits_processor(self.lm_head, hidden_states,
  File "/home/udadmin/LLM/pytorch/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/udadmin/LLM/pytorch/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/udadmin/LLM/vllm/vllm/model_executor/layers/logits_processor.py", line 75, in forward
    logits = _apply_logits_processors(logits, sampling_metadata)
  File "/home/udadmin/LLM/vllm/vllm/model_executor/layers/logits_processor.py", line 216, in _apply_logits_processors
    logits_row = logits_processor(past_tokens_ids,
  File "/home/udadmin/LLM/vllm/vllm/model_executor/guided_decoding/outlines_logits_processors.py", line 106, in __call__
    traceback.print_stack()
"""

class BaseLogitsProcessor:

    def __init__(self, guide: Guide):
        self._guide: Guide = guide
        # CFGState is used for the FSM state for CFGGuide
        self._fsm_state: DefaultDict[int, Union[int,
                                                CFGState]] = defaultdict(lambda: self._guide.initial_state)

    def __call__(self, input_ids: List[int],
                 scores: torch.Tensor) -> torch.Tensor:
        global lp
        def inner_call(input_ids, scores):
            """Use the FSM to bias the logits before sampling the next token."""
            seq_id = hash(tuple(input_ids))

            if len(input_ids) > 0:
                last_token = input_ids[-1]
                last_seq_id = hash(tuple(input_ids[:-1]))
                self._fsm_state[seq_id] = self._guide.get_next_state(
                    state=self._fsm_state[last_seq_id], token_id=last_token)
            else:
                # Note: this is a hack.
                # Lark pickling does not work properly (silent failure),
                # which breaks the RPC (which uses python pickleing).
                # We need to find a better solution.
                # On the first time this is called, we simply re-create
                # the Guide object.
                if isinstance(self._guide, CFGGuideMP):
                    self._guide = self.remake_guide()
                
                self._fsm_state[seq_id] = CFGState(
                    parser_state=self._guide.parser.parse(""), prev_token=None)

            instruction = self._guide.get_next_instruction(
                state=self._fsm_state[seq_id])

            if type(instruction) == Generate:  # noqa: E721
                allowed_tokens = instruction.tokens
            elif type(instruction) == Write:  # noqa: E721
                # TODO: support fast forward tokens
                allowed_tokens = [instruction.tokens[0]]
            else:
                raise TypeError(
                    f"Unsupported instruction type {type(instruction)}")

            mask = torch.full((scores.shape[-1], ),
                            -torch.inf,
                            device=scores.device)
            # The tokenizer may support more token ids than the model can generate,
            # eg. Llama 3.2 Vision models have an `<|image|>` token with id 128256
            # but scores.shape == torch.Size([128256])
            # Using NumPy is faster for filtering token ids
            allowed_tokens = np.array(allowed_tokens, dtype=np.int64)
            allowed_tokens = torch.tensor(allowed_tokens, device=scores.device)
            allowed_tokens = allowed_tokens.masked_select(
                allowed_tokens < scores.shape[-1])
            mask.index_fill_(0, allowed_tokens, 0)
            scores.add_(mask)
            return scores
        
        #lpw = lp(inner_call)
        #results = lpw(input_ids, scores)
        #lp.print_stats()
        return inner_call(input_ids, scores)

class RegexLogitsProcessor(BaseLogitsProcessor):

    @classmethod
    @cache()
    def _get_guide(cls, regex_string: str,
                   tokenizer: PreTrainedTokenizerBase) -> Guide:
        tokenizer = _adapt_tokenizer(tokenizer)
        return RegexGuide.from_regex(regex_string, tokenizer)

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            The model's tokenizer

        """
        super().__init__(
            RegexLogitsProcessor._get_guide(regex_string, tokenizer))
          
        self.regex_string = regex_string
    
    def remake_guide(self):
      return RegexGuide._get_guide(self.regex_string, self._guide.tokenizer)


class JSONLogitsProcessor(RegexLogitsProcessor):

    def __init__(self, schema: Union[str, Dict, BaseModel],
                 tokenizer: PreTrainedTokenizerBase,
                 whitespace_pattern: Union[str, None]):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to
            generate
        tokenizer
            The model's tokenizer
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact
            string literals)
            Example: allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        """
        if isinstance(schema, type(BaseModel)):
            schema_str = json.dumps(schema.model_json_schema())
        elif isinstance(schema, Dict):
            schema_str = json.dumps(schema)
        elif isinstance(schema, str):
            schema_str = schema
        else:
            raise ValueError(
                f"Cannot parse schema {schema}. The schema must be either "
                f"a Pydantic object, a dictionary or a string that contains "
                f"the JSON Schema specification")
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string, tokenizer)


class CFGLogitsProcessor(BaseLogitsProcessor):

    @classmethod
    @cache()
    def _get_guide(cls, cfg: str, tokenizer: PreTrainedTokenizerBase) -> Guide:
        tokenizer = _adapt_tokenizer(tokenizer)
        return CFGGuideMP(cfg, tokenizer)

    def __init__(self, cfg: str, tokenizer: PreTrainedTokenizerBase):
        """Compile the FSM that drives the context free grammar generation.

        Parameters
        ----------
        cfg
            A string that represents a context-free grammar
        tokenizer
            The model's tokenizer

        """
        super().__init__(CFGLogitsProcessor._get_guide(cfg, tokenizer))
        self._guide = self._guide.copy()
    
    def remake_guide(self):
        return CFGGuideMP(self._guide.cfg_string, self._guide.tokenizer)

@lru_cache(maxsize=32)
def _adapt_tokenizer(tokenizer: PreTrainedTokenizerBase):
    """Adapt vLLM's tokenizer to use to compile the FSM.

    The API of Outlines tokenizers is slightly different to that of
    `transformers`. The decoder of outlines, returns a list whereas
    the decode of vLLM returns an str. To sync the vLLM decoder with
    outlines internal api, the decoder should be adapted. In addition
    we need to handle the missing spaces to Llama's tokenizer to be
    able to compile FSMs for this model.

    """
    if getattr(tokenizer, "_outlines_adapted", False):
        return tokenizer

    tokenizer = TransformerTokenizer(copy.deepcopy(tokenizer))

    setattr(tokenizer, "_outlines_adapted", True)  # noqa: B010

    return tokenizer

    def change_decoder(
        decoder: Callable[[List[int]],
                          str]) -> Callable[[List[int]], List[str]]:
        """Sync vLLM's decoder with the outlines by returning list."""
      
        def new_decoder(inp_tokens: List[int]) -> List[str]:
            if (isinstance(inp_tokens, list) and len(inp_tokens) == 1
                    and isinstance(inp_tokens[0], list)):
                inp_tokens = inp_tokens[0]
            return [decoder(inp_tokens)]

        return new_decoder

    tokenizer.decode = change_decoder(tokenizer.decode)
