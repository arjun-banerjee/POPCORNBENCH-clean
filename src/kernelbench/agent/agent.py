"""
KernelAgent — multi-turn, tool-using agent for KernelBench, built on the
OpenAI Responses API.

Architecture
------------
- One agent instance handles one (problem, run) pair.
- Each call to run() returns a KernelTrajectory (full history + final result).
- Conversation state is stateless: the full `input` array is resent every turn.
  Reasoning items returned by the model flow back into `input` unchanged so
  the model sees its own chain-of-thought across turns. (The Responses API
  errors if reasoning items are dropped between turns with reasoning models.)
- Tools are declared natively via JSON schema. No XML parsing; the API
  returns structured `function_call` items and we reply with
  `function_call_output` items matched by `call_id`.
- The system prompt is passed as the top-level `instructions` parameter,
  separate from `input`.

Loop, in words:
    instructions := build_system_prompt(...)
    input := [problem_message]
    for turn in range(max_turns):
        response = client.responses.create(instructions, input, tools, ...)
        input += response.output            (preserves reasoning + tool calls)
        for fc in function_calls(response):
            result = execute_tool(fc)
            input.append(function_call_output(fc.call_id, result.output))
            if fc.name == 'submit_kernel' and result is a real KernelExecResult:
                return trajectory
        if no function_calls: break
    return trajectory
"""

from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from typing import Any

import torch
from openai import OpenAI

from kernelbench.eval import KernelExecResult
from kernelbench.agent.tools import Tool, ToolContext, ToolResult, get_tools
from kernelbench.agent.trajectory import KernelTurn, KernelTrajectory, ToolCall
from kernelbench.agent.prompt_templates import (
    build_system_prompt,
    build_problem_message,
    build_turn_warning_message,
    build_no_tool_calls_nudge,
    build_final_turn_mandatory_submit_message,
)
from kernelbench.agent.llm_utils import (
    compact_chat_tool_messages,
    compact_responses_input_items,
    extract_modelnew_from_response_items,
    extract_modelnew_kernel_from_text,
    is_requested_tokens_zero,
    llm_retry_delay_s,
    llm_usage_to_dict,
    log_trimmed_create_kwargs_diagnostics,
    maybe_sliding_window_chat,
    truncate_context_str,
)


def _strip_status(obj: Any) -> None:
    """Recursively pop `status` keys from a dumped Responses-API item.

    Azure's /openai/v1/ preview 400s if `status` is echoed back inside the
    `input` array; the public OpenAI API ignores it. Mutates in place.
    """
    if isinstance(obj, dict):
        obj.pop("status", None)
        for v in obj.values():
            _strip_status(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_status(v)


class KernelAgent:
    """
    Multi-turn, tool-using agent for a single KernelBench problem.

    The caller is responsible for constructing the OpenAI client. This keeps
    the agent agnostic to Azure/OpenAI-direct/compatible-gateway wiring — the
    caller passes in a configured `OpenAI(...)` instance with whatever
    `api_key` and `base_url` they need.

    Args:
        problem_id:         Integer problem ID.
        level:              Dataset level (1, 2, or 3).
        problem_name:       Human-readable problem name.
        ref_arch_src:       Reference PyTorch model source code.
        client:             An openai.OpenAI instance, pre-configured with
                            api_key and (optionally) base_url.
        model:              Model name to pass to client.responses.create.
        run_name:           Name for this run (used in trajectory metadata).
        tool_names:         Names of tools to enable. None = default set
                            (all except profile_kernel).
        max_turns:          Max number of LLM turns.
        max_tool_calls:     Max total tool calls across all turns.
        backend:            Kernel backend. Default "cuda".
        precision:          Computation precision. Default "fp32".
        device:             torch.device for evaluation.
        build_dir:          CUDA compile cache directory.
        num_correct_trials: Correctness trials for run_correctness / submit.
        num_perf_trials:    Timing trials for submit_kernel.
        timing_method:      Timing method for submit_kernel.
        reasoning_effort:   "minimal" | "low" | "medium" | "high" | None.
                            If set, passed to the API as reasoning.effort.
        warn_turns_remaining: Inject warning when this many turns remain.
        turn_delay_s:       Sleep between turns (for rate-limited APIs).
        llm_error_retries:  Retries (with backoff) per turn when the LLM API
                            raises or returns an unusable response; set to 1
                            to fail fast like older versions.
        verbose:            Verbose logging.
        tool_output_context_max_chars: Truncate tool outputs in LLM context (0 disables).
        reasoning_context_max_chars: Cap reasoning when compacting chat / storing.
        chat_context_tail_messages: If set, sliding window on chat history (tail size).
        llm_concurrency_semaphore: Optional semaphore acquired around each LLM HTTP call.
        omit_responses_reasoning: If true, omit the ``reasoning`` parameter on
            ``responses.create`` (for providers like xAI Grok that reject
            OpenAI-style reasoning payloads).
    """

    def __init__(
        self,
        *,
        problem_id: int,
        level: int,
        problem_name: str,
        ref_arch_src: str,
        client: OpenAI,
        model: str,
        run_name: str = "default",
        tool_names: list[str] | None = None,
        max_turns: int = 10,
        max_tool_calls: int = 30,
        backend: str = "cuda",
        precision: str = "fp32",
        device: torch.device | None = None,
        build_dir: str | None = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        timing_method: str = "cuda_event",
        reasoning_effort: str | None = None,
        warn_turns_remaining: int = 2,
        turn_delay_s: float = 0.0,
        llm_error_retries: int = 3,
        verbose: bool = False,
        api_kind: str = "openai",
        omit_responses_reasoning: bool = False,
        save_path: str | None = None,
        eval_client: Any = None,
        initial_message: str | None = None,
        tool_output_context_max_chars: int = 120_000,
        reasoning_context_max_chars: int = 16_000,
        chat_context_tail_messages: int | None = None,
        llm_concurrency_semaphore: Any = None,
    ) -> None:
        self.problem_id = problem_id
        self.level = level
        self.problem_name = problem_name
        self.ref_arch_src = ref_arch_src
        self.client = client
        self.model = model
        self.run_name = run_name
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.backend = backend
        self.precision = precision
        self.reasoning_effort = reasoning_effort
        self.warn_turns_remaining = warn_turns_remaining
        self.turn_delay_s = turn_delay_s
        self.llm_error_retries = max(1, int(llm_error_retries))
        self.verbose = verbose
        self.api_kind = api_kind
        self.omit_responses_reasoning = omit_responses_reasoning
        self.save_path = save_path
        self.tool_output_context_max_chars = max(0, int(tool_output_context_max_chars))
        self.reasoning_context_max_chars = max(0, int(reasoning_context_max_chars))
        self.chat_context_tail_messages = (
            int(chat_context_tail_messages)
            if chat_context_tail_messages is not None
            else None
        )
        self._llm_semaphore = llm_concurrency_semaphore

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Resolve tools: `tool_names` filters the registry; get_tools() ensures
        # submit_kernel is always included.
        self.tools: list[Tool] = get_tools(tool_names)
        self.tool_names_enabled: list[str] = [t.name for t in self.tools]
        self.tool_map: dict[str, Tool] = {t.name: t for t in self.tools}

        # ToolContext is constructed once and reused for every tool call this run.
        self.ctx = ToolContext(
            ref_arch_src=ref_arch_src,
            backend=backend,
            precision=precision,
            device=device,
            build_dir=build_dir,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            timing_method=timing_method,
            verbose=verbose,
            eval_client=eval_client,
        )

        # Optional override for the first user message (e.g. hw_translation prompt).
        self._initial_message = initial_message

        # Per-run mutable state.
        self._total_tool_calls: int = 0
        self._final_result: KernelExecResult | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> KernelTrajectory:
        """Execute the agent loop and return a completed KernelTrajectory."""
        if self.api_kind == "openai_chat":
            return self._run_chat_completions()
        return self._run_responses()

    def _tag(self) -> str:
        return f"[L{self.level}/P{self.problem_id}/{self.model}]"

    @contextmanager
    def _llm_acquire(self):
        sem = self._llm_semaphore
        if sem is not None:
            sem.acquire()
        try:
            yield
        finally:
            if sem is not None:
                sem.release()

    def _reset_kernel_tracking(self) -> None:
        self._last_compiled_kernel: str | None = None
        self._last_correct_kernel: str | None = None

    def _note_kernel_progress(
        self, tool_name: str, args: dict[str, Any], tool_result: ToolResult
    ) -> None:
        if tool_name not in ("compile_kernel", "run_correctness"):
            return
        if not tool_result.success:
            return
        meta = tool_result.metadata or {}
        if not meta.get("compiled"):
            return
        kc = args.get("kernel_code")
        if not isinstance(kc, str) or not kc.strip():
            return
        self._last_compiled_kernel = kc
        if meta.get("correctness"):
            self._last_correct_kernel = kc

    def _best_effort_autofinalize_after_loop(
        self, trajectory: KernelTrajectory, tag: str
    ) -> None:
        """If the model never submitted but we have a compiled kernel, submit once."""
        if self._final_result is not None:
            return
        submit_tool = self.tool_map.get("submit_kernel")
        if submit_tool is None:
            return
        kernel = self._last_correct_kernel or self._last_compiled_kernel
        if not kernel:
            return
        print(
            f"{tag} best-effort submit_kernel from last successful "
            f"{'correctness' if self._last_correct_kernel else 'compile'} …",
            flush=True,
        )
        self._total_tool_calls += 1
        tool_result = self._execute_tool(submit_tool, {"kernel_code": kernel})
        finalized = self._finalize_from_submit_result(tool_result)
        trajectory.add_turn(
            KernelTurn(
                turn_id=len(trajectory.turns),
                messages_in=[],
                response=[],
                tool_calls=[
                    ToolCall(
                        tool_name="submit_kernel",
                        args={"kernel_code": kernel},
                        result_text=tool_result.output,
                        success=tool_result.success,
                        metadata=tool_result.metadata,
                    )
                ],
                feedback_to_model="",
                llm_latency_s=0.0,
                is_final=finalized,
                submitted_kernel=kernel if finalized else None,
            )
        )

    def _install_autosave(self, trajectory: KernelTrajectory) -> None:
        """Wrap trajectory.add_turn / finish so they flush to save_path after
        every mutation. Mid-run snapshots have outcome='in_progress'; the wrapped
        finish() overwrites that with the final outcome.
        """
        if not self.save_path:
            return
        path = self.save_path
        orig_add = trajectory.add_turn
        orig_finish = trajectory.finish

        def _save():
            try:
                trajectory.save(path)
            except Exception as e:
                print(f"[Agent] mid-run save failed: {e}", flush=True)

        def add_turn(turn):
            orig_add(turn)
            if trajectory.finished_at is None:
                trajectory.outcome = "in_progress"
            _save()

        def finish(result):
            orig_finish(result)
            _save()

        trajectory.add_turn = add_turn  # type: ignore[assignment]
        trajectory.finish = finish      # type: ignore[assignment]

    def _ingest_endpoint_progress(self) -> list[dict[str, Any]]:
        """Read new entries from {trajectory_basename}_aeprogress.jsonl.

        Endpoint adapters (currently aeproxy) write one JSON line per
        intermediate program they evaluate while a chat completion is in
        flight. We pick those up at the end of each turn, convert them into
        synthetic function_call items in the trajectory's response list, and
        advance a cursor so we don't re-ingest them next turn.

        Returns the list of synthesized response items (may be empty).
        """
        if not self.save_path:
            return []
        progress_path = self.save_path.replace(
            "_trajectory.json", "_aeprogress.jsonl"
        )
        try:
            if not os.path.isfile(progress_path):
                return []
            with open(progress_path) as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[Agent] could not read {progress_path}: {e}", flush=True)
            return []

        new_lines = lines[self._endpoint_progress_cursor:]
        self._endpoint_progress_cursor = len(lines)

        items: list[dict[str, Any]] = []
        for raw in new_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                e = json.loads(raw)
            except Exception:
                continue
            kernel = e.get("kernel_src") or ""
            outcome = (
                "correct" if e.get("correct")
                else "compiled" if e.get("compiled")
                else "fail"
            )
            speedup = float(e.get("speedup") or 0.0)
            elapsed = float(e.get("elapsed_s") or 0.0)
            args_payload = {
                "kernel_code": kernel,
                "_outcome": outcome,
                "_speedup": speedup,
                "_elapsed_s": elapsed,
                "_program": e.get("program_name", ""),
            }
            if e.get("error"):
                args_payload["_error"] = e["error"]
            items.append({
                "type": "function_call",
                "name": "evaluate_ae_candidate",
                "arguments": json.dumps(args_payload),
                "call_id": f"ae_call_{e.get('idx','?')}",
            })
        return items

    def _run_responses(self) -> KernelTrajectory:
        """Original Responses-API loop."""
        tag = self._tag()
        # NOTE: install_autosave is called below, after construction.
        trajectory = KernelTrajectory(
            problem_id=self.problem_id,
            level=self.level,
            problem_name=self.problem_name,
            run_name=self.run_name,
            model_name=self.model,
            backend=self.backend,
            precision=self.precision,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            tools_enabled=self.tool_names_enabled,
        )
        self._install_autosave(trajectory)
        if self.save_path:
            trajectory.outcome = "in_progress"
            try:
                trajectory.save(self.save_path)
            except Exception as e:
                print(f"[Agent] initial save failed: {e}", flush=True)
        self._endpoint_progress_cursor = 0
        self._reset_kernel_tracking()

        instructions = build_system_prompt(
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            backend=self.backend,
            tool_names=self.tool_names_enabled,
        )
        if self._initial_message is not None:
            problem_msg = self._initial_message
        else:
            problem_msg = build_problem_message(
                ref_arch_src=self.ref_arch_src,
                backend=self.backend,
                precision=self.precision,
            )

        # The input array we resend every turn. It grows monotonically as:
        #   - the model emits reasoning / function_call items (we append all
        #     of response.output)
        #   - we append function_call_output items after each tool executes
        #   - we append our own role=user messages (problem, warnings)
        input_items: list[dict[str, Any]] = [
            {"role": "user", "content": problem_msg},
        ]

        tool_schemas = [t.to_responses_schema() for t in self.tools]

        for turn_idx in range(self.max_turns):
            # Pacing for rate-limited APIs.
            if turn_idx > 0 and self.turn_delay_s > 0:
                time.sleep(self.turn_delay_s)
            print(
                f"{tag} turn {turn_idx + 1}/{self.max_turns} → LLM call...",
                flush=True,
            )

            turns_remaining = self.max_turns - turn_idx
            tool_calls_remaining = self.max_tool_calls - self._total_tool_calls

            if turns_remaining == 1:
                input_items.append(
                    {
                        "role": "user",
                        "content": build_final_turn_mandatory_submit_message(),
                    }
                )

            if turns_remaining <= self.warn_turns_remaining and turn_idx > 0:
                input_items.append(
                    {
                        "role": "user",
                        "content": build_turn_warning_message(
                            turns_remaining,
                            tool_calls_remaining,
                        ),
                    }
                )

            if self.tool_output_context_max_chars > 0:
                compact_responses_input_items(
                    input_items,
                    tool_output_max_chars=self.tool_output_context_max_chars,
                )

            # Snapshot the input we're about to send so the trajectory can
            # replay this exact turn. Deep-copy via json round-trip so later
            # mutations of input_items don't leak in.
            messages_in_snapshot = json.loads(json.dumps(input_items))

            # --- LLM call (retry transient failures) ---
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": instructions,
                "input": input_items,
                "tools": tool_schemas,
            }
            if not self.omit_responses_reasoning:
                if self.reasoning_effort is not None:
                    create_kwargs["reasoning"] = {
                        "effort": self.reasoning_effort,
                        "summary": "auto",
                    }
                else:
                    create_kwargs["reasoning"] = {"summary": "auto"}
            if self.max_turns == 1:
                create_kwargs["tool_choice"] = "required"

            response = None
            llm_latency = 0.0
            for attempt in range(self.llm_error_retries):
                t0 = time.time()
                try:
                    with self._llm_acquire():
                        try:
                            response = self.client.responses.create(**create_kwargs)
                        except TypeError:
                            if "tool_choice" in create_kwargs:
                                create_kwargs.pop("tool_choice", None)
                                response = self.client.responses.create(
                                    **create_kwargs
                                )
                            else:
                                raise
                    llm_latency = time.time() - t0
                    break
                except Exception as e:
                    llm_latency = time.time() - t0
                    err_s = f"{type(e).__name__}: {e}"
                    if is_requested_tokens_zero(err_s):
                        print(
                            f"{tag} LLM error suggests bad request token accounting "
                            f"(Requested tokens: 0). Payload diagnostics:",
                            flush=True,
                        )
                        log_trimmed_create_kwargs_diagnostics(tag, create_kwargs)
                    if attempt + 1 >= self.llm_error_retries:
                        err_msg = err_s
                        tb = traceback.format_exc()
                        print(
                            f"{tag} LLM call FAILED after "
                            f"{self.llm_error_retries} attempt(s) on turn "
                            f"{turn_idx} (problem {self.problem_id}, "
                            f"level {self.level}):\n"
                            f"{err_msg}\n{tb}",
                            flush=True,
                        )
                        failed_turn = KernelTurn(
                            turn_id=turn_idx,
                            messages_in=messages_in_snapshot,
                            response=[],
                            feedback_to_model=f"LLM call failed: {err_msg}",
                            llm_latency_s=llm_latency,
                            is_final=False,
                        )
                        trajectory.add_turn(failed_turn)
                        trajectory.finish(self._final_result)
                        return trajectory
                    delay = llm_retry_delay_s(attempt, e)
                    print(
                        f"{tag} LLM error (attempt {attempt + 1}/"
                        f"{self.llm_error_retries}): {err_s or repr(e)}; "
                        f"retrying in {delay:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay)

            assert response is not None  # loop exits via return if all fail

            turn_usage = llm_usage_to_dict(getattr(response, "usage", None))

            # Always-on per-turn line so users can see progress in long runs.
            n_fc = sum(
                1 for it in (item.model_dump() for item in response.output)
                if it.get("type") == "function_call"
            )
            print(
                f"{tag} turn {turn_idx + 1} LLM done in {llm_latency:.1f}s "
                f"({n_fc} tool call{'s' if n_fc != 1 else ''})",
                flush=True,
            )

            # Serialize the model's output items so we can both (a) resend
            # them to the API next turn and (b) store them in the trajectory.
            # model_dump() gives us plain dicts, which is what both consumers
            # want.
            #
            # Azure's /openai/v1/ preview rejects the `status` field that the
            # SDK echoes back on reasoning/function_call items
            # (`Unknown parameter: 'input[N].status'`). The public OpenAI API
            # tolerates it. Strip it on the way back in to keep both happy.
            response_items: list[dict[str, Any]] = []
            for item in response.output:
                d = item.model_dump()
                _strip_status(d)
                response_items.append(d)
            input_items.extend(response_items)

            if self.verbose:
                n_fc = sum(
                    1 for it in response_items if it.get("type") == "function_call"
                )

                print(
                    f"\n[Agent] Turn {turn_idx} "
                    f"({llm_latency:.1f}s, {len(response_items)} output items, "
                    f"{n_fc} function calls)"
                )

                for it in response_items:
                    if it.get("type") == "function_call":
                        fc = it.get("function_call", {})
                        print(f"[fn] {fc.get('name')}({fc.get('arguments')})")

            # --- Execute tool calls ---
            function_calls = [
                it for it in response_items if it.get("type") == "function_call"
            ]

            executed_tool_calls: list[ToolCall] = []
            is_final = False
            submitted_kernel: str | None = None

            if not function_calls:
                if turn_idx >= self.max_turns - 1:
                    fb_kernel = (
                        extract_modelnew_from_response_items(response_items)
                        or self._last_correct_kernel
                        or self._last_compiled_kernel
                    )
                    executed_fb: list[ToolCall] = []
                    is_final_fb = False
                    submitted_fb: str | None = None
                    if fb_kernel and "submit_kernel" in self.tool_map:
                        print(
                            f"{tag} turn {turn_idx + 1}: no tool calls — "
                            f"fallback submit_kernel (ModelNew guard) …",
                            flush=True,
                        )
                        self._total_tool_calls += 1
                        tr = self._execute_tool(
                            self.tool_map["submit_kernel"],
                            {"kernel_code": fb_kernel},
                        )
                        is_final_fb = self._finalize_from_submit_result(tr)
                        submitted_fb = fb_kernel if is_final_fb else None
                        executed_fb.append(
                            ToolCall(
                                tool_name="submit_kernel",
                                args={"kernel_code": fb_kernel},
                                result_text=tr.output,
                                success=tr.success,
                                metadata=tr.metadata,
                            )
                        )
                    trajectory.add_turn(
                        KernelTurn(
                            turn_id=turn_idx,
                            messages_in=messages_in_snapshot,
                            response=response_items,
                            tool_calls=executed_fb,
                            feedback_to_model="",
                            llm_latency_s=llm_latency,
                            is_final=is_final_fb,
                            submitted_kernel=submitted_fb,
                            llm_usage=turn_usage,
                        )
                    )
                    if is_final_fb and self.verbose:
                        print(
                            f"[Agent] Final submission on turn {turn_idx} "
                            f"(no-tools fallback).",
                            flush=True,
                        )
                    break
                trajectory.add_turn(
                    KernelTurn(
                        turn_id=turn_idx,
                        messages_in=messages_in_snapshot,
                        response=response_items,
                        tool_calls=[],
                        feedback_to_model="",
                        llm_latency_s=llm_latency,
                        is_final=False,
                        llm_usage=turn_usage,
                    )
                )
                input_items.append(
                    {"role": "user", "content": build_no_tool_calls_nudge()}
                )
                print(
                    f"{tag} turn {turn_idx + 1}: no tool calls — nudging",
                    flush=True,
                )
                continue

            for fc in function_calls:
                # Enforce the total tool-call budget. We still need to reply
                # with a function_call_output for every function_call so the
                # API doesn't error on the next turn — so synthesize a
                # budget-exceeded output for any call beyond the cap.
                if self._total_tool_calls >= self.max_tool_calls:
                    output_text = (
                        "Tool call limit reached. No further tool calls will "
                        "be executed this run."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=fc.get("name", "?"),
                            args={},
                            result_text=output_text,
                            success=False,
                            metadata={"skipped": "tool_call_limit_reached"},
                        )
                    )
                    continue

                tool_name = fc.get("name", "")
                raw_args = fc.get("arguments", "{}")
                try:
                    args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else (raw_args or {})
                    )
                except json.JSONDecodeError as e:
                    output_text = (
                        f"Tool call arguments could not be parsed as JSON: {e}. "
                        "Please re-issue the call with valid JSON arguments."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args={"_raw": raw_args},
                            result_text=output_text,
                            success=False,
                            metadata={"error": "invalid_json_arguments"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                if tool_name not in self.tool_map:
                    output_text = (
                        f"Unknown tool '{tool_name}'. Available tools: "
                        f"{', '.join(self.tool_names_enabled)}."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args=args,
                            result_text=output_text,
                            success=False,
                            metadata={"error": "unknown_tool"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                tool = self.tool_map[tool_name]
                _t_tool = time.time()
                tool_result = self._execute_tool(tool, args)
                _tool_dt = time.time() - _t_tool
                print(
                    f"{tag}   {tool_name} {'OK' if tool_result.success else 'FAIL'} "
                    f"({_tool_dt:.1f}s)",
                    flush=True,
                )
                self._total_tool_calls += 1

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc["call_id"],
                        "output": tool_result.output,
                    }
                )

                # Save full args (including kernel source) into the trajectory
                # so reviewers can see exactly what the model produced.
                executed_tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=dict(args),
                        result_text=tool_result.output,
                        success=tool_result.success,
                        metadata=tool_result.metadata,
                    )
                )
                self._note_kernel_progress(tool_name, args, tool_result)

                # submit_kernel finalizes the run, but only if it returned a
                # real KernelExecResult. Transient errors (e.g. lock file
                # contention) leave `compiled`/`correctness` absent from
                # metadata — in those cases let the model retry.
                if tool_name == "submit_kernel":
                    submitted_kernel = args.get("kernel_code")
                    if self._finalize_from_submit_result(tool_result):
                        is_final = True
                        break

            if not is_final:
                is_final, auto_kernel = self._maybe_autofinalize_last_turn(
                    turn_idx=turn_idx,
                    executed_tool_calls=executed_tool_calls,
                )
                if auto_kernel:
                    submitted_kernel = auto_kernel

            turn = KernelTurn(
                turn_id=turn_idx,
                messages_in=messages_in_snapshot,
                response=response_items,
                tool_calls=executed_tool_calls,
                feedback_to_model="",
                llm_latency_s=llm_latency,
                is_final=is_final,
                submitted_kernel=submitted_kernel,
                llm_usage=turn_usage,
            )
            trajectory.add_turn(turn)

            if is_final:
                if self.verbose:
                    print(f"[Agent] Final submission on turn {turn_idx}.")
                break

        self._best_effort_autofinalize_after_loop(trajectory, tag)
        trajectory.finish(self._final_result)
        return trajectory

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _execute_tool(self, tool: Tool, args: dict[str, Any]) -> ToolResult:
        """Execute a single tool, catching unexpected exceptions."""
        try:
            return tool.execute(self.ctx, **args)
        except Exception as e:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                output=(
                    f"{tool.name} FAILED: unexpected error during tool execution.\n"
                    f"{type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                ),
                metadata={"unexpected_error": str(e)},
            )

    def _finalize_from_submit_result(self, tool_result: ToolResult) -> bool:
        """Record a submit_kernel result as the run's final outcome when valid."""
        meta = tool_result.metadata or {}
        if "compiled" not in meta or "correctness" not in meta:
            return False
        try:
            self._final_result = KernelExecResult(**meta)
        except Exception:
            self._final_result = None
        return True

    def _maybe_autofinalize_last_turn(
        self,
        *,
        turn_idx: int,
        executed_tool_calls: list[ToolCall],
    ) -> tuple[bool, str | None]:
        """On the last turn, auto-submit after successful compile or correctness.

        Avoids `error` when the model runs out of turns right after a good
        compile_kernel or run_correctness without calling submit_kernel.
        """
        if turn_idx != self.max_turns - 1 or not executed_tool_calls:
            return False, None

        last_call = executed_tool_calls[-1]
        meta = last_call.metadata or {}
        kernel_code = last_call.args.get("kernel_code")
        if (
            not last_call.success
            or not isinstance(kernel_code, str)
            or not kernel_code
            or not meta.get("compiled")
        ):
            return False, None
        if last_call.tool_name == "run_correctness" and not meta.get("correctness"):
            return False, None
        if last_call.tool_name not in ("run_correctness", "compile_kernel"):
            return False, None

        submit_tool = self.tool_map.get("submit_kernel")
        if submit_tool is None:
            return False, None

        tag = self._tag()
        _t_tool = time.time()
        tool_result = self._execute_tool(submit_tool, {"kernel_code": kernel_code})
        _tool_dt = time.time() - _t_tool
        print(
            f"{tag}   submit_kernel {'OK' if tool_result.success else 'FAIL'} "
            f"({_tool_dt:.1f}s) [auto-finalize]",
            flush=True,
        )
        self._total_tool_calls += 1
        executed_tool_calls.append(
            ToolCall(
                tool_name="submit_kernel",
                args={"kernel_code": kernel_code},
                result_text=tool_result.output,
                success=tool_result.success,
                metadata=tool_result.metadata,
            )
        )

        return self._finalize_from_submit_result(tool_result), kernel_code

    # -----------------------------------------------------------------------
    # Chat Completions code path
    # -----------------------------------------------------------------------

    def _run_chat_completions(self) -> KernelTrajectory:
        """Agent loop that uses the OpenAI Chat Completions API.

        Used for models behind endpoints that don't speak the Responses API
        (e.g. DeepSeek-R1, Llama-Maverick, Kimi via Azure AI Inference).

        Differences vs. _run_responses():
          - State is a list of messages: {role, content, ...}.
          - Tool calls come back on the assistant message as `tool_calls=[...]`,
            and we reply with role="tool" messages keyed by tool_call_id.
          - Tool schemas use the nested {"type":"function","function":{...}}
            shape rather than the flat Responses shape.
          - No `reasoning` items; if a model returns reasoning_content (e.g.
            DeepSeek-R1) we capture it for the trajectory but don't resend it.
        """
        tag = self._tag()
        trajectory = KernelTrajectory(
            problem_id=self.problem_id,
            level=self.level,
            problem_name=self.problem_name,
            run_name=self.run_name,
            model_name=self.model,
            backend=self.backend,
            precision=self.precision,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            tools_enabled=self.tool_names_enabled,
        )
        self._install_autosave(trajectory)
        if self.save_path:
            trajectory.outcome = "in_progress"
            try:
                trajectory.save(self.save_path)
            except Exception as e:
                print(f"[Agent] initial save failed: {e}", flush=True)
        self._endpoint_progress_cursor = 0
        self._reset_kernel_tracking()

        instructions = build_system_prompt(
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            backend=self.backend,
            tool_names=self.tool_names_enabled,
        )
        if self._initial_message is not None:
            problem_msg = self._initial_message
        else:
            problem_msg = build_problem_message(
                ref_arch_src=self.ref_arch_src,
                backend=self.backend,
                precision=self.precision,
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": problem_msg},
        ]

        # Convert each tool's flat Responses schema into the nested Chat shape.
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self.tools
        ]

        for turn_idx in range(self.max_turns):
            if turn_idx > 0 and self.turn_delay_s > 0:
                time.sleep(self.turn_delay_s)
            print(
                f"{tag} turn {turn_idx + 1}/{self.max_turns} → LLM call...",
                flush=True,
            )

            turns_remaining = self.max_turns - turn_idx
            tool_calls_remaining = self.max_tool_calls - self._total_tool_calls

            if turns_remaining == 1:
                messages.append(
                    {
                        "role": "user",
                        "content": build_final_turn_mandatory_submit_message(),
                    }
                )

            if turns_remaining <= self.warn_turns_remaining and turn_idx > 0:
                messages.append(
                    {
                        "role": "user",
                        "content": build_turn_warning_message(
                            turns_remaining,
                            tool_calls_remaining,
                        ),
                    }
                )

            if self.tool_output_context_max_chars > 0:
                compact_chat_tool_messages(
                    messages,
                    tool_output_max_chars=self.tool_output_context_max_chars,
                )
            if self.chat_context_tail_messages is not None:
                maybe_sliding_window_chat(
                    messages,
                    keep_tail=self.chat_context_tail_messages,
                )

            messages_in_snapshot = json.loads(json.dumps(messages))

            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": tool_schemas,
                "tool_choice": "required" if self.max_turns == 1 else "auto",
            }
            if self.save_path:
                create_kwargs["extra_body"] = {
                    "popcornbench_run_meta": {
                        "run_name": self.run_name,
                        "level": self.level,
                        "problem_id": self.problem_id,
                        "model": self.model,
                        "turn_id": turn_idx,
                        "trajectory_dir": os.path.dirname(self.save_path),
                        "trajectory_basename": os.path.splitext(
                            os.path.basename(self.save_path)
                        )[0],
                    }
                }

            response = None
            llm_latency = 0.0
            for attempt in range(self.llm_error_retries):
                t0 = time.time()
                try:
                    with self._llm_acquire():
                        response = self.client.chat.completions.create(**create_kwargs)
                    if not response.choices:
                        raise ValueError("chat completion returned no choices")
                    _chk = response.choices[0].message
                    if _chk is None:
                        raise ValueError("chat completion choice has no message")
                    llm_latency = time.time() - t0
                    break
                except Exception as e:
                    llm_latency = time.time() - t0
                    err_s = f"{type(e).__name__}: {e}"
                    if is_requested_tokens_zero(err_s):
                        print(
                            f"{tag} chat LLM error suggests bad request token "
                            f"accounting (Requested tokens: 0). Payload diagnostics:",
                            flush=True,
                        )
                        log_trimmed_create_kwargs_diagnostics(tag, create_kwargs)
                    if attempt + 1 >= self.llm_error_retries:
                        err_msg = err_s
                        tb = traceback.format_exc()
                        print(
                            f"{tag} chat LLM FAILED after "
                            f"{self.llm_error_retries} attempt(s) on turn "
                            f"{turn_idx} (problem {self.problem_id}, "
                            f"level {self.level}):\n"
                            f"{err_msg}\n{tb}",
                            flush=True,
                        )
                        trajectory.add_turn(
                            KernelTurn(
                                turn_id=turn_idx,
                                messages_in=messages_in_snapshot,
                                response=[],
                                feedback_to_model=f"LLM call failed: {err_msg}",
                                llm_latency_s=llm_latency,
                                is_final=False,
                            )
                        )
                        trajectory.finish(self._final_result)
                        return trajectory
                    delay = llm_retry_delay_s(attempt, e)
                    print(
                        f"{tag} chat LLM error (attempt {attempt + 1}/"
                        f"{self.llm_error_retries}): {err_s or repr(e)}; "
                        f"retrying in {delay:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay)

            assert response is not None
            turn_usage = llm_usage_to_dict(getattr(response, "usage", None))
            _n_fc_chat = len(response.choices[0].message.tool_calls or [])
            print(
                f"{tag} turn {turn_idx + 1} LLM done in {llm_latency:.1f}s "
                f"({_n_fc_chat} tool call{'s' if _n_fc_chat != 1 else ''})",
                flush=True,
            )

            choice = response.choices[0]
            asst = choice.message
            assert asst is not None
            asst_content = asst.content or ""
            reasoning = getattr(asst, "reasoning_content", None) or ""
            raw_tool_calls = list(asst.tool_calls or [])

            # DeepSeek-R1 (and similar) emit reasoning inline as <think>...</think>
            # in the content. Split those out into the reasoning slot and strip
            # them from the visible content so they're not resent next turn.
            if "<think>" in asst_content:
                import re as _re
                think_blocks = _re.findall(r"<think>(.*?)</think>", asst_content, _re.S)
                if think_blocks:
                    reasoning = (reasoning + "\n" + "\n".join(think_blocks)).strip()
                    asst_content = _re.sub(
                        r"<think>.*?</think>", "", asst_content, flags=_re.S
                    ).strip()

            # Build the assistant message that goes back into `messages` for
            # the next turn. Chat Completions requires tool_calls (and ids) to
            # be echoed back in the assistant message; otherwise the role=tool
            # replies have nothing to anchor to.
            asst_msg: dict[str, Any] = {
                "role": "assistant",
                "content": asst_content,
            }
            if raw_tool_calls:
                asst_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in raw_tool_calls
                ]
            messages.append(asst_msg)

            # Build a Responses-shaped record for the trajectory (so the HTML
            # report doesn't have to know about both API shapes).
            response_items: list[dict[str, Any]] = []
            if reasoning:
                if self.reasoning_context_max_chars > 0:
                    reasoning = truncate_context_str(
                        reasoning,
                        self.reasoning_context_max_chars,
                        "reasoning",
                    )
                response_items.append(
                    {"type": "reasoning", "summary": [{"text": reasoning}]}
                )
            # Endpoint adapters (notably aeproxy) write per-step progress to
            # {trajectory_basename}_aeprogress.jsonl while the chat completion
            # is in flight. Ingest those entries here so the saved trajectory
            # records every intermediate program AE produced — not just the
            # final tool call.
            response_items.extend(self._ingest_endpoint_progress())
            if asst_content:
                response_items.append(
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": asst_content}],
                    }
                )
            for tc in raw_tool_calls:
                response_items.append(
                    {
                        "type": "function_call",
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                        "call_id": tc.id,
                    }
                )

            executed_tool_calls: list[ToolCall] = []
            is_final = False
            submitted_kernel: str | None = None

            if not raw_tool_calls:
                if turn_idx >= self.max_turns - 1:
                    fb_kernel = (
                        extract_modelnew_kernel_from_text(asst_content)
                        or self._last_correct_kernel
                        or self._last_compiled_kernel
                    )
                    executed_fb: list[ToolCall] = []
                    is_final_fb = False
                    submitted_fb: str | None = None
                    if fb_kernel and "submit_kernel" in self.tool_map:
                        print(
                            f"{tag} turn {turn_idx + 1}: no tool calls — "
                            f"fallback submit_kernel (ModelNew guard) …",
                            flush=True,
                        )
                        self._total_tool_calls += 1
                        tr = self._execute_tool(
                            self.tool_map["submit_kernel"],
                            {"kernel_code": fb_kernel},
                        )
                        is_final_fb = self._finalize_from_submit_result(tr)
                        submitted_fb = fb_kernel if is_final_fb else None
                        executed_fb.append(
                            ToolCall(
                                tool_name="submit_kernel",
                                args={"kernel_code": fb_kernel},
                                result_text=tr.output,
                                success=tr.success,
                                metadata=tr.metadata,
                            )
                        )
                    trajectory.add_turn(
                        KernelTurn(
                            turn_id=turn_idx,
                            messages_in=messages_in_snapshot,
                            response=response_items,
                            tool_calls=executed_fb,
                            feedback_to_model="",
                            llm_latency_s=llm_latency,
                            is_final=is_final_fb,
                            submitted_kernel=submitted_fb,
                            llm_usage=turn_usage,
                        )
                    )
                    if is_final_fb and self.verbose:
                        print(
                            f"[Agent/chat] Final submission on turn {turn_idx} "
                            f"(no-tools fallback).",
                            flush=True,
                        )
                    break
                trajectory.add_turn(
                    KernelTurn(
                        turn_id=turn_idx,
                        messages_in=messages_in_snapshot,
                        response=response_items,
                        tool_calls=[],
                        feedback_to_model="",
                        llm_latency_s=llm_latency,
                        is_final=False,
                        llm_usage=turn_usage,
                    )
                )
                messages.append(
                    {"role": "user", "content": build_no_tool_calls_nudge()}
                )
                print(
                    f"{tag} turn {turn_idx + 1}: no tool calls — nudging",
                    flush=True,
                )
                continue

            for tc in raw_tool_calls:
                tool_name = tc.function.name or ""
                raw_args = tc.function.arguments or "{}"

                if self._total_tool_calls >= self.max_tool_calls:
                    output_text = (
                        "Tool call limit reached. No further tool calls will "
                        "be executed this run."
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": output_text}
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args={},
                            result_text=output_text,
                            success=False,
                            metadata={"skipped": "tool_call_limit_reached"},
                        )
                    )
                    continue

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError as e:
                    output_text = (
                        f"Tool call arguments could not be parsed as JSON: {e}. "
                        "Please re-issue the call with valid JSON arguments."
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": output_text}
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args={"_raw": raw_args},
                            result_text=output_text,
                            success=False,
                            metadata={"error": "invalid_json_arguments"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                if tool_name not in self.tool_map:
                    output_text = (
                        f"Unknown tool '{tool_name}'. Available tools: "
                        f"{', '.join(self.tool_names_enabled)}."
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": output_text}
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args=args,
                            result_text=output_text,
                            success=False,
                            metadata={"error": "unknown_tool"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                tool = self.tool_map[tool_name]
                _t_tool = time.time()
                tool_result = self._execute_tool(tool, args)
                _tool_dt = time.time() - _t_tool
                print(
                    f"{tag}   {tool_name} {'OK' if tool_result.success else 'FAIL'} "
                    f"({_tool_dt:.1f}s)",
                    flush=True,
                )
                self._total_tool_calls += 1

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result.output,
                    }
                )

                executed_tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=dict(args),
                        result_text=tool_result.output,
                        success=tool_result.success,
                        metadata=tool_result.metadata,
                    )
                )
                self._note_kernel_progress(tool_name, args, tool_result)

                if tool_name == "submit_kernel":
                    submitted_kernel = args.get("kernel_code")
                    if self._finalize_from_submit_result(tool_result):
                        is_final = True
                        break

            if not is_final:
                is_final, auto_kernel = self._maybe_autofinalize_last_turn(
                    turn_idx=turn_idx,
                    executed_tool_calls=executed_tool_calls,
                )
                if auto_kernel:
                    submitted_kernel = auto_kernel

            trajectory.add_turn(
                KernelTurn(
                    turn_id=turn_idx,
                    messages_in=messages_in_snapshot,
                    response=response_items,
                    tool_calls=executed_tool_calls,
                    feedback_to_model="",
                    llm_latency_s=llm_latency,
                    is_final=is_final,
                    submitted_kernel=submitted_kernel,
                    llm_usage=turn_usage,
                )
            )

            if is_final:
                if self.verbose:
                    print(f"[Agent/chat] Final submission on turn {turn_idx}.")
                break

        self._best_effort_autofinalize_after_loop(trajectory, tag)
        trajectory.finish(self._final_result)
        return trajectory
