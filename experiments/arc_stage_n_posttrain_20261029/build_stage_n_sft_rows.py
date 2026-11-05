"""Four-operator selective-loss SFT row builder for Stage-N post-training.

Supports the four production inference operators of Stage-LM:
1. `direct_generate`  — Clean multi-turn solve from public task JSON (identity/transpose/rot90).
2. `code_repair`      — In-conversation recovery after an observed tool/demo/submission failure.
3. `near_miss_repair` — Repair of `[PRIOR_WAVE_NEAR_MISS_CANDIDATE]` supplied in USER prompt.
4. `adjudicate`       — Evidence-based `[DISAGREEMENT_AND_OVERFIT_ADJUDICATION]` across competing
                        candidates (including verified unified replacement and validated
                        insufficient-evidence clean abstention).
"""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from stage_n_catalog import STAGE_N_DATA_REVISION, TOOLS, sha256_text

ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n"
THINK_SUFFIX = "\n</think>\n\n"
IM_END = "<|im_end|>"
IM_END_ID = 248046
MIN_REASONING_CHARS = 32
STAGE_N_ACCEPTED_MAX_TOKENS = 45000
STAGE_N_HARD_MAX_TOKENS = 49152

STAGE_N_OPERATORS = frozenset({
    "direct_generate",
    "code_repair",
    "near_miss_repair",
    "adjudicate",
})

NEAR_MISS_MARKER = "[PRIOR_WAVE_NEAR_MISS_CANDIDATE]"
ADJUDICATE_MARKER = "[DISAGREEMENT_AND_OVERFIT_ADJUDICATION]"

# Reject any trajectory whose assistant reasoning/content leaks offline gold or hidden verifiers.
HINDSIGHT_FORBIDDEN_PATTERNS = re.compile(
    r"\b(?:sealed[_\s-]*gold|hidden[_\s-]*(?:test|gold|verifier|answer|output)|"
    r"ground[_\s-]*truth[_\s-]*(?:output|grid|answer)|re[_-]?arc[_\s-]*generator)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Turn:
    message_index: int
    kind: str
    after_failure: bool
    after_all_pass_verify: bool
    reasoning_chars: int
    tool_call_ids: tuple[str, ...]


def arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool_arguments_not_object")
    return parsed


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in (content or []) if isinstance(b, dict))


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    for message in result:
        if message["role"] == "assistant":
            message["reasoning_content"] = message.get("reasoning_content") or ""
            message["content"] = message.get("content") or ""
            for call in message.get("tool_calls") or []:
                call["function"]["arguments"] = arguments(call["function"]["arguments"])
    return result


def ensure_system_message(messages: list[dict[str, Any]], default: str = "") -> list[dict[str, Any]]:
    messages = copy.deepcopy(messages)
    system = next(
        (
            message_text(m.get("content"))
            for m in messages
            if m.get("role") == "system" and message_text(m.get("content")).strip()
        ),
        default,
    )
    return [{"role": "system", "content": system}] + [m for m in messages if m.get("role") != "system"]


def pi_events_to_messages(lines: list[Any], *, system: str) -> list[dict[str, Any]]:
    events = [json.loads(line) if isinstance(line, str) else line for line in lines]
    request = next((e for e in events if e.get("type") == "request"), None)
    request_messages = (request or {}).get("payload", {}).get("messages", [])
    first = request_messages[0] if request_messages else {}
    request_system = message_text(first.get("content")) if first.get("role") == "system" else ""
    event_system = next(
        (
            message_text(e["message"].get("content"))
            for e in events
            if e.get("type") == "message_end"
            and e.get("message", {}).get("role") == "system"
            and message_text(e["message"].get("content")).strip()
        ),
        "",
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": request_system if request_system.strip() else event_system or system}
    ]
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event["message"]
        role = message["role"]
        blocks = message.get("content") or []
        if role == "user":
            content = blocks if isinstance(blocks, str) else "".join(b.get("text", "") for b in blocks)
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": arguments(b["arguments"])},
                }
                for b in blocks
                if b.get("type") == "toolCall"
            ]
            messages.append(
                {
                    "role": "assistant",
                    "reasoning_content": "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking"),
                    "content": "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
                    "tool_calls": calls,
                }
            )
        elif role == "toolResult":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message["toolCallId"],
                    "content": "".join(b.get("text", "") for b in blocks),
                }
            )
    return messages


def parse_feedback(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def printed_demo_reports(stdout: Any) -> list[dict[str, Any]]:
    if not isinstance(stdout, str):
        return []
    reports: list[dict[str, Any]] = []
    candidates = [stdout.strip()] + [s.strip() for s in stdout.splitlines()]
    seen: set[str] = set()
    for text in candidates:
        if text in seen or not text.startswith(("{", "[")):
            continue
        seen.add(text)
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                obj = json.loads(text)
            except ValueError:
                continue
        items = obj if isinstance(obj, list) else [obj]
        reports.extend(x for x in items if isinstance(x, dict) and isinstance(x.get("demos"), list))
    return reports


def demo_reports(feedback: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if isinstance(feedback.get("demos"), list):
        reports.append(feedback)
    reports += feedback.get("verification_reports") or []
    reports += printed_demo_reports(feedback.get("stdout"))
    return reports


def execution_broken(feedback: dict[str, Any]) -> bool:
    """True when the tool execution itself errored or a `submit_program` was rejected."""
    if feedback.get("ok") is False or feedback.get("eligible") is False:
        return True
    return feedback.get("kind") in {"syntax", "runtime", "timeout", "contract", "infrastructure", "deadline"}


def has_demo_mismatch(feedback: dict[str, Any]) -> bool:
    if feedback.get("verification_summary", {}).get("failed_reports", 0) > 0:
        return True
    return any(
        d.get("kind") != "pass"
        for report in demo_reports(feedback)
        for d in (report.get("demos") or [])
    )


def has_all_pass_demo_report(feedback: dict[str, Any]) -> bool:
    if execution_broken(feedback) or has_demo_mismatch(feedback):
        return False
    reps = demo_reports(feedback)
    return bool(reps) and any(
        bool(r.get("demos")) and all(d.get("kind") == "pass" for d in r["demos"])
        for r in reps
    )


def infer_prompt_operator(messages: list[dict[str, Any]]) -> str | None:
    first_user = next((message_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    if ADJUDICATE_MARKER in first_user:
        return "adjudicate"
    if NEAR_MISS_MARKER in first_user:
        return "near_miss_repair"
    return None


def contains_hindsight_leak(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for field in ("reasoning_content", "content"):
            text = msg.get(field) or ""
            if isinstance(text, str) and HINDSIGHT_FORBIDDEN_PATTERNS.search(text):
                return True
    return False


def classify_assistant_turns(
    messages: list[dict[str, Any]],
    *,
    prompt_operator: str | None = None,
    allow_abstention: bool = False,
) -> list[Turn]:
    """Classify every assistant turn with operator-aware diagnostic & abstention semantics."""
    op = prompt_operator if prompt_operator is not None else infer_prompt_operator(messages)
    diagnostic_operator = op in {"near_miss_repair", "adjudicate"}
    turns: list[Turn] = []
    seen_failure = False
    seen_all_pass_verify = False

    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    last_assistant_idx = assistant_indices[-1] if assistant_indices else -1

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        calls = message.get("tool_calls") or []
        receipts: dict[str, dict[str, Any] | None] = {}
        for following in messages[index + 1 :]:
            if following["role"] == "assistant":
                break
            if following["role"] == "tool":
                cid = str(following.get("tool_call_id") or "")
                if cid in receipts:
                    raise ValueError("duplicate_tool_receipt")
                receipts[cid] = parse_feedback(following.get("content"))

        if not calls:
            # Validated terminal abstention is allowed ONLY on the final turn of an `adjudicate` episode
            # after at least one clean/diagnostic Python experiment has been executed.
            body_text = (message.get("content") or "").strip()
            has_prior_experiment = any(
                t.kind in {"clean_inspection", "diagnostic_experiment"} for t in turns
            )
            if (
                allow_abstention
                and op == "adjudicate"
                and index == last_assistant_idx
                and has_prior_experiment
                and len(body_text) >= 16
            ):
                kind = "insufficient_evidence_stop"
            else:
                kind = "unverified"
            turns.append(
                Turn(
                    index,
                    kind,
                    seen_failure,
                    seen_all_pass_verify,
                    len((message.get("reasoning_content") or "").strip()),
                    (),
                )
            )
            continue

        feedbacks = [receipts.get(str(call.get("id") or "")) for call in calls]
        complete = bool(calls) and all(f is not None for f in feedbacks)
        broken = any(execution_broken(f) for f in feedbacks if f is not None)
        demo_mismatch = any(has_demo_mismatch(f) for f in feedbacks if f is not None)
        is_python_only = complete and all(c["function"]["name"] == "python" for c in calls)
        is_verified_submit = (
            complete
            and not broken
            and not demo_mismatch
            and any(
                call["function"]["name"] == "submit_program" and (fb or {}).get("eligible") is True
                for call, fb in zip(calls, feedbacks)
            )
        )

        if not complete:
            kind = "unverified"
        elif broken:
            kind = "failed_attempt"
        elif is_verified_submit:
            if op == "near_miss_repair":
                kind = "verified_near_miss_repair"
            elif op == "adjudicate":
                kind = "verified_adjudication"
            else:
                kind = "verified_resolution" if seen_failure else "direct_solve"
        elif is_python_only:
            unobserved = any(
                re.search(r"\bverify_demos\s*\(", arguments(c["function"]["arguments"]).get("code", ""))
                and not demo_reports(fb or {})
                for c, fb in zip(calls, feedbacks)
            )
            if unobserved:
                kind = "unverified"
            elif demo_mismatch:
                # Key Stage-N fix (`gpt-6-astra` Section 2):
                # In `near_miss_repair` and `adjudicate`, a clean (`ok: True`) Python experiment
                # that reproduces a supplied candidate's demo mismatch falsifies a hypothesis and
                # IS useful diagnostic supervision, whereas in `direct_generate` / `code_repair`
                # it marks a failed proposed hypothesis.
                kind = "diagnostic_experiment" if diagnostic_operator else "failed_attempt"
            else:
                kind = "clean_inspection"
        else:
            kind = "failed_attempt" if demo_mismatch else "unverified"

        turns.append(
            Turn(
                index,
                kind,
                seen_failure,
                seen_all_pass_verify,
                len((message.get("reasoning_content") or "").strip()),
                tuple(str(c.get("id") or "") for c in calls),
            )
        )
        if kind == "failed_attempt":
            seen_failure = True
        if is_python_only and not broken and any(has_all_pass_demo_report(fb or {}) for fb in feedbacks):
            seen_all_pass_verify = True
        if kind == "failed_attempt":
            # A later failure invalidates an earlier passing verify report.
            seen_all_pass_verify = False

    return turns


def train_action(turn: Turn, operator_type: str) -> bool:
    if turn.kind in {"failed_attempt", "unverified"}:
        return False
    # Never train extra `python` exploration AFTER `verify_demos` already passed all demos:
    # SYSTEM explicitly instructs to submit once `verify_demos` passes (`gpt-6-astra` Section 2).
    if turn.after_all_pass_verify and turn.kind in {"clean_inspection", "diagnostic_experiment"}:
        return False
    if operator_type == "direct_generate":
        return not turn.after_failure and turn.kind in {"clean_inspection", "direct_solve"}
    if operator_type == "code_repair":
        return turn.after_failure and turn.kind in {"clean_inspection", "verified_resolution"}
    if operator_type == "near_miss_repair":
        return turn.kind in {"clean_inspection", "diagnostic_experiment", "verified_near_miss_repair"}
    if operator_type == "adjudicate":
        return turn.kind in {
            "clean_inspection",
            "diagnostic_experiment",
            "verified_adjudication",
            "insufficient_evidence_stop",
        }
    raise ValueError(f"unknown_operator_type:{operator_type}")


def train_thinking(turn: Turn, operator_type: str) -> bool:
    return train_action(turn, operator_type) and turn.reasoning_chars >= MIN_REASONING_CHARS


def split_assistant_turn(turn_body: str) -> tuple[str, str]:
    idx = turn_body.find(THINK_SUFFIX)
    if idx < 0:
        raise ValueError("missing_think_close_in_assistant_turn")
    cut = idx + len(THINK_SUFFIX)
    thinking_part, action_part = turn_body[:cut], turn_body[cut:]
    if not action_part or not action_part.endswith(IM_END):
        raise ValueError("empty_or_unclosed_assistant_action")
    return thinking_part, action_part


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return list(encoded.ids if hasattr(encoded, "ids") else encoded)


def coalesce_segments(segments: list[dict[str, Any]], tokenizer: Any = None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in segments:
        if not segment["text"]:
            continue
        if merged and merged[-1]["train"] == segment["train"]:
            merged[-1]["text"] += segment["text"]
        else:
            merged.append(dict(segment))
    if tokenizer is not None:
        whole = "".join(s["text"] for s in merged)
        parts = [x for s in merged for x in token_ids(tokenizer, s["text"])]
        if parts != token_ids(tokenizer, whole):
            raise ValueError("segment_tokenization_mismatch")
    return merged


def split_prompt_segments_by_operator(
    prompt: str,
    turn_train_flags: list[tuple[bool, bool]],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    position = 0
    turn_number = 0
    while True:
        start = prompt.find(ASSISTANT_PREFIX, position)
        if start < 0:
            if position < len(prompt):
                segments.append({"text": prompt[position:], "train": False})
            break
        body = start + len(ASSISTANT_PREFIX)
        segments.append({"text": prompt[position:body], "train": False})
        if body == len(prompt):
            break
        end = prompt.find(IM_END, body)
        if end < 0 or turn_number >= len(turn_train_flags):
            raise ValueError("assistant_boundary_mismatch")
        end += len(IM_END)
        thinking_flag, action_flag = turn_train_flags[turn_number]
        turn_body = prompt[body:end]
        if thinking_flag == action_flag:
            segments.append({"text": turn_body, "train": bool(thinking_flag)})
        else:
            thinking_part, action_part = split_assistant_turn(turn_body)
            segments.append({"text": thinking_part, "train": bool(thinking_flag)})
            segments.append({"text": action_part, "train": bool(action_flag)})
        turn_number += 1
        position = end
    if turn_number != len(turn_train_flags) or "".join(s["text"] for s in segments) != prompt:
        raise ValueError("assistant_boundary_mismatch")
    return coalesce_segments(segments, tokenizer)


TERMINAL_SOLVE_KINDS = frozenset({
    "direct_solve",
    "verified_resolution",
    "verified_near_miss_repair",
    "verified_adjudication",
    "insufficient_evidence_stop",
})


def build_four_operator_segments(
    messages: list[dict[str, Any]],
    template: Any,
    tokenizer: Any,
    operator_type: str,
    *,
    allow_abstention: bool = False,
    reject_post_verify_exploration: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[Turn], int]:
    if operator_type not in STAGE_N_OPERATORS:
        raise ValueError(f"unsupported_operator_type:{operator_type}")
    if contains_hindsight_leak(messages):
        raise ValueError("hindsight_gold_rationale_leak_rejected")

    turns = classify_assistant_turns(
        messages,
        prompt_operator=operator_type if operator_type in {"near_miss_repair", "adjudicate"} else None,
        allow_abstention=allow_abstention,
    )
    solved = [t for t in turns if t.kind in TERMINAL_SOLVE_KINDS]
    if not solved:
        raise ValueError("no_verified_terminal_turn")
    final = solved[-1]
    history = messages[: final.message_index + 1]
    selected = [t for t in turns if t.message_index <= final.message_index]
    if not train_action(final, operator_type):
        raise ValueError("final_turn_not_supervisable")
    if operator_type == "direct_generate" and any(t.kind == "failed_attempt" for t in selected):
        raise ValueError("direct_row_contains_failure_history")
    if reject_post_verify_exploration and any(
        t.after_all_pass_verify and t.kind in {"clean_inspection", "diagnostic_experiment"}
        for t in selected[:-1]
    ):
        raise ValueError("post_verify_exploration_conflicts_with_system")

    options = dict(tools=TOOLS, enable_thinking=True, preserve_thinking=True, reasoning_effort="medium")
    prompt = template.render(messages=history[:-1], add_generation_prompt=True, **options)
    full = template.render(messages=history, add_generation_prompt=False, **options)
    if not full.startswith(prompt) or not full.endswith(IM_END + "\n"):
        raise ValueError("template_prefix_or_suffix_mismatch")
    final_body = full[len(prompt) : -1]
    flags = [(train_thinking(t, operator_type), train_action(t, operator_type)) for t in selected[:-1]]
    segments = split_prompt_segments_by_operator(prompt, flags, tokenizer)
    if train_thinking(final, operator_type):
        completion = final_body
    else:
        final_thinking_part, completion = split_assistant_turn(final_body)
        segments = coalesce_segments(segments + [{"text": final_thinking_part, "train": False}], tokenizer)
    prompt_text = "".join(s["text"] for s in segments)
    prompt_ids, completion_ids = token_ids(tokenizer, prompt_text), token_ids(tokenizer, completion)
    if not completion_ids or completion_ids[-1] != IM_END_ID:
        raise ValueError("unexpected_im_end_id")
    if prompt_ids + completion_ids != token_ids(tokenizer, full[:-1]):
        raise ValueError("prompt_completion_tokenization_mismatch")
    return segments, completion_ids, selected, len(prompt_ids)


def build_stage_n_row(
    task_id: str,
    messages: list[dict[str, Any]],
    template: Any,
    tokenizer: Any,
    *,
    eligible: bool,
    exact_match: bool,
    source: str,
    split_group: str,
    is_pure_arc2: bool,
    coordinate_frame: str = "identity",
    operator_override: str | None = None,
    allow_abstention: bool = False,
    reject_post_verify_exploration: bool = False,
    adjudication_meta: dict[str, Any] | None = None,
    sample_weight: float = 1.0,
    max_length: int = STAGE_N_ACCEPTED_MAX_TOKENS,
    compacted_turns: int = 0,
) -> dict[str, Any]:
    if not (0.0 < float(sample_weight) <= 1.25):
        raise ValueError(f"sample_weight_out_of_bounds:{sample_weight}")
    if eligible is not True or exact_match is not True:
        raise ValueError("requires_external_eligible_and_exact_match")
    if coordinate_frame not in {"identity", "transpose", "rot90", "flip_h"}:
        raise ValueError(f"unsupported_coordinate_frame:{coordinate_frame}")

    messages = normalize_messages(ensure_system_message(messages))
    system_text = messages[0]["content"]
    system_chars = len(system_text.strip())
    if system_chars != 2414:
        raise ValueError(f"unexpected_system_prompt_chars:{system_chars}")

    prompt_op = infer_prompt_operator(messages)
    if operator_override is not None:
        if prompt_op is not None and prompt_op != operator_override:
            raise ValueError(f"operator_mismatch_with_prompt:{operator_override}!={prompt_op}")
        operator = operator_override
    elif prompt_op is not None:
        operator = prompt_op
    else:
        turns_probe = classify_assistant_turns(messages)
        operator = "code_repair" if any(t.kind == "failed_attempt" for t in turns_probe) else "direct_generate"

    segments, completion, selected, prompt_length = build_four_operator_segments(
        messages,
        template,
        tokenizer,
        operator,
        allow_abstention=allow_abstention,
        reject_post_verify_exploration=reject_post_verify_exploration,
    )
    total = prompt_length + len(completion)
    if total >= max_length:
        raise ValueError(f"overlength_stage_n_row:{total}>={max_length}")

    prefix = "".join(s["text"] for s in segments).split(IM_END, 1)[0].removeprefix("<|im_start|>system\n")
    if (
        len(prefix) != 3876
        or not prefix.endswith(system_text)
        or "<tools>" not in prefix
        or '"name": "python"' not in prefix
        or '"name": "submit_program"' not in prefix
    ):
        raise ValueError(f"incomplete_rendered_system_header:{len(prefix)}")

    terminal_kind = selected[-1].kind
    row: dict[str, Any] = {
        "schema": "v7-trace-sft-row-v1",
        "data_revision": STAGE_N_DATA_REVISION,
        "task_id": task_id,
        "operator_type": operator,
        "terminal_kind": terminal_kind,
        "source": source,
        "split_group": split_group,
        "is_pure_arc2": bool(is_pure_arc2),
        "coordinate_frame": coordinate_frame,
        "sample_weight": float(sample_weight),
        "query_correct": True,
        "last_prompt_tokens": prompt_length,
        "total_tokens": total,
        "system_prompt_chars": system_chars,
        "rendered_system_chars": len(prefix),
        "system_prompt_sha256": sha256_text(system_text),
        "rendered_system_sha256": sha256_text(prefix),
        "segments": segments,
        "completion": {"token_ids": completion, "finish_reason": "stop"},
        "turn_classification": [
            asdict(t)
            | {
                "train": train_action(t, operator),
                "train_thinking": train_thinking(t, operator),
                "train_action": train_action(t, operator),
            }
            for t in selected
        ],
        "compacted_turns": int(compacted_turns),
        "synthetic_reasoning": False,
    }
    if adjudication_meta is not None:
        row["adjudication_meta"] = copy.deepcopy(adjudication_meta)
    return row


def try_replay_verified_prefix_compaction(
    task: dict[str, Any],
    messages: list[dict[str, Any]],
    template: Any,
    tokenizer: Any,
    *,
    max_tokens: int = STAGE_N_ACCEPTED_MAX_TOKENS,
    trusted_fixture_mode: bool = False,
) -> tuple[list[dict[str, Any]], int] | None:
    """Safely prune a genuinely removable early prefix ONLY when fresh-workspace replay matches.

    Follows `gpt-6-astra` Section 1:
    1. Only drops contiguous early assistant+tool turn blocks starting right after `messages[1]` (USER).
    2. Rejects if any variable assigned solely in dropped `python` calls is referenced in retained code
       or retained `<think>` text.
    3. Replays every retained `python` call and final `submit_program` in a fresh `Workspace` and
       requires identical `ok`/`eligible` outcomes and zero `NameError`s.
    """
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root / "src") not in _sys.path:
        _sys.path.insert(0, str(repo_root / "src"))
    from arc_feedback_harness.workspace import Limits, Workspace, check_candidate

    norm = normalize_messages(ensure_system_message(messages))
    if len(norm) < 6 or norm[0]["role"] != "system" or norm[1]["role"] != "user":
        return None

    # Group `[assistant_msg, tool_msg_1, ...]` blocks after `[system, user]`.
    blocks: list[list[dict[str, Any]]] = []
    idx = 2
    while idx < len(norm):
        if norm[idx]["role"] != "assistant":
            return None
        blk = [norm[idx]]
        idx += 1
        while idx < len(norm) and norm[idx]["role"] == "tool":
            blk.append(norm[idx])
            idx += 1
        blocks.append(blk)

    if len(blocks) <= 2:
        return None

    def _assigned_names(code: str) -> set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        return names

    def _loaded_names(code: str) -> set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }

    options = dict(tools=TOOLS, enable_thinking=True, preserve_thinking=True, reasoning_effort="medium")
    limits = Limits(trusted_fixture_mode=trusted_fixture_mode)

    # Try dropping the smallest prefix `k = 1 .. len(blocks) - 2` that fits `< max_tokens`.
    for drop_k in range(1, len(blocks) - 1):
        dropped = blocks[:drop_k]
        retained = blocks[drop_k:]
        candidate_msgs = [norm[0], norm[1], *[m for blk in retained for m in blk]]
        rendered = template.render(messages=candidate_msgs, add_generation_prompt=False, **options)
        if len(token_ids(tokenizer, rendered[:-1])) >= max_tokens:
            continue

        # Check static dependency & reasoning reference independence from dropped prefix.
        dropped_defs: set[str] = set()
        for blk in dropped:
            for call in blk[0].get("tool_calls") or []:
                if call["function"]["name"] == "python":
                    dropped_defs |= _assigned_names(call["function"]["arguments"].get("code", ""))

        retained_defs_so_far: set[str] = set()
        dependency_leak = False
        for blk in retained:
            asst = blk[0]
            think_text = (asst.get("reasoning_content") or "") + "\n" + (asst.get("content") or "")
            # Reject explicit backward turn references or dropped-only variable names in prose.
            unresolved = dropped_defs - retained_defs_so_far
            for sym in unresolved:
                if len(sym) >= 3 and re.search(rf"\b{re.escape(sym)}\b", think_text):
                    dependency_leak = True
                    break
            if dependency_leak:
                break
            for call in asst.get("tool_calls") or []:
                code = call["function"]["arguments"].get("code", "")
                if call["function"]["name"] == "python":
                    used = _loaded_names(code)
                    if used & (dropped_defs - retained_defs_so_far):
                        dependency_leak = True
                        break
                    retained_defs_so_far |= _assigned_names(code)
            if dependency_leak:
                break
        if dependency_leak:
            continue

        # Fresh-workspace execution replay of every retained tool call.
        ws = Workspace(task, limits)
        replay_ok = True
        try:
            step_n = 0
            for blk in retained:
                asst = blk[0]
                tool_map = {
                    str(m.get("tool_call_id")): parse_feedback(m.get("content"))
                    for m in blk[1:]
                }
                for call in asst.get("tool_calls") or []:
                    step_n += 1
                    cid = str(call.get("id"))
                    orig_fb = tool_map.get(cid)
                    if orig_fb is None:
                        replay_ok = False
                        break
                    fn = call["function"]["name"]
                    code = call["function"]["arguments"].get("code", "")
                    if fn == "python":
                        rep = ws.run(code, f"compact_py_{step_n}", mode="exec")
                        if bool(rep.get("ok")) != bool(orig_fb.get("ok")):
                            replay_ok = False
                            break
                        if not rep.get("ok") and "NameError" in str(rep.get("error", "")):
                            replay_ok = False
                            break
                    elif fn == "submit_program":
                        chk = check_candidate(task, code, f"compact_chk_{step_n}", limits)
                        all_pass = (
                            bool(chk.get("ok"))
                            and len(chk.get("demos") or []) == len(task["train"])
                            and all(
                                d.get("ok") and d.get("grid") == p["output"]
                                for d, p in zip(chk["demos"], task["train"])
                            )
                            and all(q.get("ok") for q in (chk.get("queries") or []))
                        )
                        if all_pass != bool(orig_fb.get("eligible")):
                            replay_ok = False
                            break
                if not replay_ok:
                    break
        finally:
            ws.close()

        if replay_ok:
            return candidate_msgs, drop_k

    return None


def load_renderer(tokenizer_dir: str | Path) -> tuple[Any, Any]:
    from jinja2 import Environment
    from tokenizers import Tokenizer

    root = Path(tokenizer_dir)
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if tokenizer.token_to_id(IM_END) != IM_END_ID:
        raise ValueError("tokenizer_identity_mismatch")
    env = Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    env.filters["items"] = lambda value: (json.loads(value) if isinstance(value, str) else value).items()

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    env.globals["raise_exception"] = raise_exception
    return env.from_string((root / "chat_template.jinja").read_text(encoding="utf-8")), tokenizer
