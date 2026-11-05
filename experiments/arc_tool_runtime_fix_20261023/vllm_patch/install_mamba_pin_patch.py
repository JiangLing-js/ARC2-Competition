"""Pin hashed Mamba/GDN prompt-boundary state snapshots while their request runs.

vLLM v0.30.0, ``--mamba-cache-mode align`` + fine-grained prefix matching.
Without this patch the prompt-end state snapshot of a running request is
released to the evictable free queue at decode step 1 (CoW ``cow_block`` is not
in ``req_blocks``; retired hashed full-block states are freed by
``_remove_blocks_in_range`` / ``remove_skipped_blocks``).  Long decode turns
outlive the free-queue turnover, so the next tool continuation misses the whole
prefix (L4 audit 2026-10-24: previous turn > 600 s -> 93 % zero-hit).

The patch keeps one extra reference on those hashed snapshot blocks (at most
``ARC_MAMBA_PIN_MAX_PER_REQ`` per request and Mamba group, newest kept) and
returns them to the free queue when the request's blocks are popped for free,
i.e. right before the continuation arrives.  ``ARC_MAMBA_PIN=0`` disables it at
runtime without reinstalling.

Usage (before spawning the vLLM server):
    python install_mamba_pin_patch.py /opt/arc_pi_venv RECEIPT.json
"""
import hashlib
import json
from pathlib import Path
import sys

MARKER = "# ---- ARC_MAMBA_PIN_PATCH v1 ----"
EXPECTED_VLLM = "0.30.0"

# Exact upstream text the patch re-implements or depends on.  Installation
# fails closed if any anchor is missing, so a different vLLM build is never
# silently patched.
ANCHORS = [
    "class MambaManager(SingleTypeKVCacheManager):",
    "from vllm.utils.math_utils import cdiv",
    # CoW branch of MambaManager.allocate_new_blocks (running request)
    "                        self.block_pool.move_block_hashes(source_block, cow_block)\n"
    "                        self._pending_cow_copies.append((source_block, cow_block))\n"
    "                        source_block.ref_cnt += 1\n",
    # MambaManager._remove_blocks_in_range align branch (re-implemented below)
    "        blocks = self.req_to_blocks.get(request_id, [])\n"
    "        first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))\n"
    "        last_block = min(last_block, len(blocks))\n"
    "        if first_block >= last_block:\n"
    "            return\n"
    "        freed: list[KVCacheBlock] = []\n"
    "        # Mamba prefill leaves null gaps between states awaiting retirement.\n"
    "        for i in range(last_block - 1, first_block - 1, -1):\n"
    "            if blocks[i].is_null:\n"
    "                continue\n"
    "            freed.append(blocks[i])\n"
    "            blocks[i] = self._null_block\n"
    "        if freed:\n"
    "            self.block_pool.free_blocks(freed)\n"
    "        self._num_retired_blocks[request_id] = last_block\n",
    # MambaManager.remove_skipped_blocks align branch (re-implemented below)
    "            last_state_block_idx = self.last_state_block_idx.get(request_id)\n",
    "            if (\n"
    "                last_state_block_idx is not None\n"
    "                and last_state_block_idx\n"
    "                < cdiv(processed_computed_tokens, self.block_size) - 1\n"
    "            ):\n"
    "                blocks = self.req_to_blocks[request_id]\n"
    "                if blocks[last_state_block_idx] != self._null_block:\n"
    "                    self.block_pool.free_blocks([blocks[last_state_block_idx]])\n"
    "                    blocks[last_state_block_idx] = self._null_block\n",
    # MambaManager.pop_blocks_for_free align bookkeeping
    "            self._producer_partial_tail_reqs.pop(request_id, None)\n",
]

PATCH = MARKER + r'''
import os as _arc_os
import time as _arc_time

_ARC_PIN_ENABLED = _arc_os.environ.get("ARC_MAMBA_PIN", "1") != "0"
_ARC_PIN_MAX = max(1, int(_arc_os.environ.get("ARC_MAMBA_PIN_MAX_PER_REQ", "1")))
_ARC_PIN_DECODE = _arc_os.environ.get("ARC_MAMBA_PIN_DECODE", "1") != "0"
_ARC_PIN_DECODE_MAX = max(0, int(_arc_os.environ.get("ARC_MAMBA_PIN_DECODE_MAX", "2")))
_ARC_DUAL_PIN_MIN_FREE = max(0, int(_arc_os.environ.get("ARC_MAMBA_DUAL_PIN_MIN_FREE", "12")))
_ARC_HANDOFF_MAX = max(0, int(_arc_os.environ.get("ARC_MAMBA_HANDOFF_MAX", "32")))
_ARC_HANDOFF_PER_REQ = max(1, int(_arc_os.environ.get("ARC_MAMBA_HANDOFF_PER_REQ", "1")))
_ARC_HANDOFF_TTL_S = float(_arc_os.environ.get("ARC_MAMBA_HANDOFF_TTL_S", "180.0"))
_ARC_HANDOFF_RECENT_GRACE_S = float(_arc_os.environ.get("ARC_MAMBA_HANDOFF_RECENT_GRACE_S", "30.0"))
_ARC_HANDOFF_MIN_FREE = max(0, int(_arc_os.environ.get("ARC_MAMBA_HANDOFF_MIN_FREE", "6")))
_ARC_PIN_STATS = {
    "managers": 0,
    "cow_pins": 0,
    "retired_pins": 0,
    "decode_retired_pins": 0,
    "tail_pins": 0,
    "decode_tail_pins": 0,
    "decode_cache_calls": 0,
    "overflow_released": 0,
    "decode_overflow_released": 0,
    "pressure_decode_released": 0,
    "pressure_prompt_released": 0,
    "request_released": 0,
    "handoff_superseded_released": 0,
    "handoff_held": 0,
    "handoff_reused": 0,
    "handoff_expired": 0,
    "handoff_expired_ttl": 0,
    "handoff_evicted_cap": 0,
    "handoff_evicted_pressure": 0,
    "pinned_now": 0,
    "pinned_peak": 0,
    "handoff_now": 0,
}
_ARC_PIN_LOG_EVERY = int(_arc_os.environ.get("ARC_MAMBA_PIN_LOG_EVERY", "25"))

_arc_orig_init = MambaManager.__init__
_arc_orig_allocate_new_blocks = MambaManager.allocate_new_blocks
_arc_orig_remove_blocks_in_range = MambaManager._remove_blocks_in_range
_arc_orig_remove_skipped_blocks = MambaManager.remove_skipped_blocks
_arc_orig_pop_blocks_for_free = MambaManager.pop_blocks_for_free
_arc_orig_cache_blocks = getattr(MambaManager, "cache_blocks", None)
_arc_orig_add_local_computed = getattr(MambaManager, "add_local_computed_blocks", None)
_arc_orig_get_num_blocks_to_allocate = getattr(MambaManager, "get_num_blocks_to_allocate", None)


def _arc_pin_log(event):
    total = (
        _ARC_PIN_STATS["cow_pins"]
        + _ARC_PIN_STATS["retired_pins"]
        + _ARC_PIN_STATS["tail_pins"]
    )
    if event == "init" or (_ARC_PIN_LOG_EVERY > 0 and total % _ARC_PIN_LOG_EVERY == 0):
        logger.info(
            "ARC_MAMBA_PIN %s %s max_per_req=%d decode=%s decode_max=%d",
            event,
            _ARC_PIN_STATS,
            _ARC_PIN_MAX,
            _ARC_PIN_DECODE,
            _ARC_PIN_DECODE_MAX,
        )


def _arc_block_tokens(self, block, idx=None):
    tok = getattr(block, "block_hash_num_tokens", None)
    if tok is not None and tok > 0:
        return int(tok)
    if idx is not None:
        return (int(idx) + 1) * int(self.block_size)
    return 0


def _arc_sync_req_pins(self, request_id):
    p_pins = self._arc_prompt_pins.get(request_id) or []
    d_pins = self._arc_decode_pins.get(request_id) or []
    combined = list(p_pins) + list(d_pins)
    if combined:
        self._arc_pinned[request_id] = combined
    else:
        self._arc_pinned.pop(request_id, None)
    return combined


def _arc_is_decode_tok(prompt_tokens, tok):
    return prompt_tokens is not None and prompt_tokens > 0 and tok > prompt_tokens


def _arc_has_prompt_pin(self, request_id, prompt_tokens):
    if prompt_tokens is None or prompt_tokens <= 0:
        return False
    p_pins = self._arc_prompt_pins.get(request_id)
    if not p_pins:
        return False
    pin_tokens = self._arc_pin_tokens.get(request_id, {})
    return any(0 < pin_tokens.get(id(b), 0) <= prompt_tokens for b in p_pins)


def _arc_max_bucket_tokens(self, request_id, decode_bucket=False):
    bucket = (
        self._arc_decode_pins.get(request_id)
        if decode_bucket
        else self._arc_prompt_pins.get(request_id)
    )
    if not bucket:
        return 0
    pin_tokens = self._arc_pin_tokens.get(request_id, {})
    return max((pin_tokens.get(id(b), 0) for b in bucket), default=0)


def _arc_pool_managers(self):
    bp = getattr(self, "block_pool", None)
    if bp is None:
        return [self]
    mgrs = [m for m in _ARC_MANAGERS if getattr(m, "block_pool", None) is bp]
    return mgrs if mgrs else [self]


def _arc_pool_under_dual_pressure(self, extra_needed=0):
    pool_total = int(getattr(self.block_pool, "num_gpu_blocks", 0) or 0)
    if pool_total <= 0 or _ARC_DUAL_PIN_MIN_FREE <= 0:
        return False
    if not hasattr(self.block_pool, "get_num_free_blocks"):
        return False
    return int(self.block_pool.get_num_free_blocks()) - int(extra_needed) < _ARC_DUAL_PIN_MIN_FREE


def _arc_reclaim_pool_pins(self, target_free=None):
    """Reclaim superseded older decode & prompt pins across all Mamba managers sharing `block_pool`."""
    pool_total = int(getattr(self.block_pool, "num_gpu_blocks", 0) or 0)
    goal = _ARC_DUAL_PIN_MIN_FREE if target_free is None else max(0, int(target_free))
    if pool_total <= 0 or goal <= 0 or not hasattr(self.block_pool, "get_num_free_blocks"):
        return
    if int(self.block_pool.get_num_free_blocks()) >= goal:
        return
    mgrs = _arc_pool_managers(self)

    def _best_common_decode_tok(rid):
        common = None
        for m in mgrs:
            d_list = m._arc_decode_pins.get(rid) or ()
            p_map = m._arc_pin_tokens.get(rid, {})
            toks = {
                int(p_map.get(id(b), 0))
                for b in d_list
                if b is not None and not b.is_null and b.block_hash is not None and int(p_map.get(id(b), 0)) > 0
            }
            if not toks:
                return 0
            common = toks if common is None else (common & toks)
            if not common:
                return 0
        return max(common) if common else 0

    cand_rids = []
    seen_rids = set()
    for m in mgrs:
        for rid in list(m._arc_decode_pins.keys()):
            if rid not in seen_rids:
                seen_rids.add(rid)
                common_tok = _best_common_decode_tok(rid)
                if common_tok > 0:
                    cand_rids.append((rid, common_tok))

    # 1) First trim any decode pins strictly older than the newest common aligned decode boundary
    for rid, common_tok in cand_rids:
        if int(self.block_pool.get_num_free_blocks()) >= goal:
            return
        to_free = []
        for m in mgrs:
            d_pins = m._arc_decode_pins.get(rid)
            if d_pins and len(d_pins) > 1:
                pin_tok = m._arc_pin_tokens.get(rid, {})
                older = [b for b in d_pins if int(pin_tok.get(id(b), 0)) < common_tok]
                if older and len(older) < len(d_pins):
                    older_ids = {id(b) for b in older}
                    m._arc_decode_pins[rid] = [b for b in d_pins if id(b) not in older_ids]
                    for b in older:
                        pin_tok.pop(id(b), None)
                    _ARC_PIN_STATS["pinned_now"] -= len(older)
                    _ARC_PIN_STATS["pressure_decode_released"] += len(older)
                    _arc_sync_req_pins(m, rid)
                    to_free.extend(older)
        if to_free:
            self.block_pool.free_blocks(to_free)

    # 2) Next release superseded prompt pins on requests that hold a common aligned decode pin in every group
    for rid, _common_tok in cand_rids:
        if int(self.block_pool.get_num_free_blocks()) >= goal:
            return
        to_free = []
        for m in mgrs:
            old_p = m._arc_prompt_pins.pop(rid, None)
            if old_p:
                pin_tok = m._arc_pin_tokens.get(rid, {})
                for b in old_p:
                    pin_tok.pop(id(b), None)
                _ARC_PIN_STATS["pinned_now"] -= len(old_p)
                _ARC_PIN_STATS["pressure_prompt_released"] += len(old_p)
                _arc_sync_req_pins(m, rid)
                to_free.extend(old_p)
        if to_free:
            self.block_pool.free_blocks(to_free)


def _arc_pin_add(self, request_id, block, boundary_tokens=0, is_decode=False):
    """Hold one reference on ``block`` (already counted by the caller) in prompt or decode bucket."""
    pin_tokens = self._arc_pin_tokens.setdefault(request_id, {})
    all_pins = self._arc_pinned.get(request_id, ())
    if any(b is block for b in all_pins):
        if boundary_tokens > 0:
            pin_tokens[id(block)] = max(pin_tokens.get(id(block), 0), boundary_tokens)
        return False

    if is_decode:
        bucket = self._arc_decode_pins.setdefault(request_id, [])
        # Under elevated physical KV occupancy on real GPU pools, trim to 1 active decode pin
        # and release older prompt/decode pins pool-wide once aligned across all Mamba groups.
        under_pressure = _arc_pool_under_dual_pressure(self)
        cap = 1 if (under_pressure and _ARC_PIN_DECODE_MAX > 0) else _ARC_PIN_DECODE_MAX
    else:
        bucket = self._arc_prompt_pins.setdefault(request_id, [])
        under_pressure = False
        cap = _ARC_PIN_MAX

    if cap <= 0:
        self.block_pool.free_blocks([block])
        return False

    bucket.append(block)
    if boundary_tokens > 0:
        pin_tokens[id(block)] = int(boundary_tokens)
    # Order bucket by boundary token position so newest snapshots always survive overflow.
    bucket.sort(key=lambda b: pin_tokens.get(id(b), 0))
    _ARC_PIN_STATS["pinned_now"] += 1
    _ARC_PIN_STATS["pinned_peak"] = max(_ARC_PIN_STATS["pinned_peak"], _ARC_PIN_STATS["pinned_now"])

    if len(bucket) > cap:
        overflow = bucket[: len(bucket) - cap]
        del bucket[: len(overflow)]
        for b in overflow:
            pin_tokens.pop(id(b), None)
        _ARC_PIN_STATS["pinned_now"] -= len(overflow)
        _ARC_PIN_STATS["overflow_released"] += len(overflow)
        if is_decode:
            _ARC_PIN_STATS["decode_overflow_released"] += len(overflow)
        self.block_pool.free_blocks(overflow)

    _arc_sync_req_pins(self, request_id)
    if is_decode and under_pressure:
        _arc_reclaim_pool_pins(self, _ARC_DUAL_PIN_MIN_FREE)
    return True


def _arc_divert_hashed(self, request_id, blocks, indices=None):
    """Split retiring blocks: prompt & decode snapshots go to independent buckets."""
    to_free = []
    prompt_tokens = self._arc_prompt_tokens.get(request_id)
    for pos, block in enumerate(blocks):
        idx = indices[pos] if indices is not None else None
        tok = _arc_block_tokens(self, block, idx)
        if not block.is_null and block.block_hash is not None:
            is_decode = _arc_is_decode_tok(prompt_tokens, tok)
            if is_decode and (not _ARC_PIN_DECODE or _ARC_PIN_DECODE_MAX <= 0):
                to_free.append(block)
                continue
            _ARC_PIN_STATS["retired_pins"] += 1
            if is_decode:
                _ARC_PIN_STATS["decode_retired_pins"] += 1
            _arc_pin_add(self, request_id, block, tok, is_decode=is_decode)
            _arc_pin_log("retired_decode" if is_decode else "retired")
        else:
            to_free.append(block)
    return to_free


def _arc_handoff_supported(self):
    return _ARC_HANDOFF_MAX > 0 and hasattr(self.block_pool, "get_num_free_blocks")


def _arc_sweep_handoff(self, keep_blocks=(), min_free=None):
    mgrs = _arc_pool_managers(self)
    if not any(getattr(m, "_arc_handoff", None) for m in mgrs):
        return
    now = _arc_time.monotonic()
    keep_ids = {id(b) for b in keep_blocks if b is not None}
    expired_ttl = []
    evicted_cap = []
    for m in mgrs:
        handoff_list = getattr(m, "_arc_handoff", None)
        if not handoff_list:
            continue
        remaining = []
        for block, ts in handoff_list:
            if id(block) not in keep_ids and (now - ts >= _ARC_HANDOFF_TTL_S or block.block_hash is None):
                expired_ttl.append(block)
            else:
                remaining.append((block, ts))
        if len(remaining) > _ARC_HANDOFF_MAX:
            need_cap = len(remaining) - _ARC_HANDOFF_MAX
            kept_cap = []
            for block, ts in remaining:
                if need_cap > 0 and id(block) not in keep_ids:
                    evicted_cap.append(block)
                    need_cap -= 1
                else:
                    kept_cap.append((block, ts))
            remaining = kept_cap
        m._arc_handoff = remaining

    # Free TTL/cap blocks BEFORE measuring free_blocks so pressure eviction sees real physical availability.
    pre_freed = expired_ttl + evicted_cap
    if pre_freed:
        _ARC_PIN_STATS["handoff_expired_ttl"] += len(expired_ttl)
        _ARC_PIN_STATS["handoff_evicted_cap"] += len(evicted_cap)
        _ARC_PIN_STATS["handoff_expired"] += len(pre_freed)
        self.block_pool.free_blocks(pre_freed)

    target_free = _ARC_HANDOFF_MIN_FREE if min_free is None else max(0, int(min_free))
    evicted_pressure = []
    if target_free > 0 and hasattr(self.block_pool, "get_num_free_blocks"):
        free_now = int(self.block_pool.get_num_free_blocks())
        if free_now < target_free:
            pool_total = int(getattr(self.block_pool, "num_gpu_blocks", 0) or 0)
            reclaim_goal = (
                max(target_free, _ARC_DUAL_PIN_MIN_FREE)
                if (min_free is None and pool_total > 0)
                else target_free
            )
            protect_recent = (
                pool_total > 0
                and free_now >= 2
                and _ARC_HANDOFF_RECENT_GRACE_S > 0.0
            )
            protected_ids = set(keep_ids)
            if protect_recent:
                for m in mgrs:
                    h_list = getattr(m, "_arc_handoff", ())
                    for blk, ts in h_list[-8:]:
                        if now - ts < _ARC_HANDOFF_RECENT_GRACE_S:
                            protected_ids.add(id(blk))

            while int(self.block_pool.get_num_free_blocks()) < reclaim_goal:
                oldest_m = None
                oldest_idx = -1
                oldest_ts = None
                for m in mgrs:
                    h_list = getattr(m, "_arc_handoff", ())
                    for idx, (blk, ts) in enumerate(h_list):
                        if id(blk) in protected_ids:
                            continue
                        if oldest_ts is None or ts < oldest_ts:
                            oldest_ts = ts
                            oldest_m = m
                            oldest_idx = idx
                        break
                if oldest_m is None or oldest_idx < 0:
                    break
                blk, _ts = oldest_m._arc_handoff.pop(oldest_idx)
                evicted_pressure.append(blk)
                self.block_pool.free_blocks([blk])

    if evicted_pressure:
        _ARC_PIN_STATS["handoff_evicted_pressure"] += len(evicted_pressure)
        _ARC_PIN_STATS["handoff_expired"] += len(evicted_pressure)
    _ARC_PIN_STATS["handoff_now"] = sum(len(getattr(m, "_arc_handoff", ())) for m in _ARC_MANAGERS)


_ARC_MANAGERS = []


def _arc_init(self, *args, **kwargs):
    _arc_orig_init(self, *args, **kwargs)
    self._arc_pinned = {}
    self._arc_prompt_pins = {}
    self._arc_decode_pins = {}
    self._arc_pin_tokens = {}
    self._arc_prompt_tokens = {}
    self._arc_pending_keep = {}
    self._arc_pending_need = {}
    self._arc_handoff = []
    _ARC_MANAGERS.append(self)
    _ARC_PIN_STATS["managers"] += 1
    _arc_pin_log("init")


def _arc_cache_blocks(self, request, num_tokens, retention_interval=None, *, replay_boundaries=()):
    prompt_tokens = getattr(request, "num_prompt_tokens", None)
    if prompt_tokens is not None and prompt_tokens > 0:
        self._arc_prompt_tokens[request.request_id] = int(prompt_tokens)
    if (
        _ARC_PIN_DECODE
        and self.mamba_cache_mode == "align"
        and prompt_tokens is not None
        and num_tokens > prompt_tokens
    ):
        retention_interval = None
        _ARC_PIN_STATS["decode_cache_calls"] += 1
    return _arc_orig_cache_blocks(
        self,
        request,
        num_tokens,
        retention_interval=retention_interval,
        replay_boundaries=replay_boundaries,
    )


def _arc_allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    if self.mamba_cache_mode != "align":
        return _arc_orig_allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model)
    if _arc_pool_under_dual_pressure(self):
        _arc_reclaim_pool_pins(self, _ARC_DUAL_PIN_MIN_FREE)
    _arc_sweep_handoff(self, min_free=_ARC_HANDOFF_MIN_FREE)
    running = request_id in self._allocated_block_reqs
    before = len(self._pending_cow_copies)
    returned = _arc_orig_allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model)
    if running and len(self._pending_cow_copies) > before:
        req_blocks = self.req_to_blocks.get(request_id, [])
        prompt_tokens = self._arc_prompt_tokens.get(request_id)
        for _source, cow_block in self._pending_cow_copies[before:]:
            # Running-request CoW: the boundary hash moved to cow_block, which
            # is outside req_blocks and would drop to ref 0 after the copy.
            if cow_block.block_hash is not None and all(b is not cow_block for b in req_blocks):
                tok = _arc_block_tokens(self, cow_block) or (prompt_tokens or 0)
                is_decode = _arc_is_decode_tok(prompt_tokens, tok)
                if is_decode and (not _ARC_PIN_DECODE or _ARC_PIN_DECODE_MAX <= 0):
                    continue
                cow_block.ref_cnt += 1
                _ARC_PIN_STATS["cow_pins"] += 1
                _arc_pin_add(self, request_id, cow_block, tok, is_decode=is_decode)
                _arc_pin_log("cow")
    return returned


def _arc_remove_blocks_in_range(self, request_id, first_block, last_block):
    if self.mamba_cache_mode != "align":
        return _arc_orig_remove_blocks_in_range(self, request_id, first_block, last_block)
    blocks = self.req_to_blocks.get(request_id, [])
    first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))
    last_block = min(last_block, len(blocks))
    if first_block >= last_block:
        return
    retiring_blocks = []
    retiring_indices = []
    # Traverse oldest -> newest so _arc_divert_hashed pins the newest block last
    # and evicts older blocks; unpinned blocks are freed in reverse order.
    for i in range(first_block, last_block):
        if blocks[i].is_null:
            continue
        retiring_blocks.append(blocks[i])
        retiring_indices.append(i)
        blocks[i] = self._null_block
    freed = _arc_divert_hashed(self, request_id, retiring_blocks, retiring_indices)
    if freed:
        self.block_pool.free_blocks(list(reversed(freed)))
    self._num_retired_blocks[request_id] = last_block


def _arc_remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
    if self.mamba_cache_mode != "align":
        return _arc_orig_remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens)
    assert isinstance(self.kv_cache_spec, MambaSpec)
    if num_prompt_tokens is not None and num_prompt_tokens > 0:
        self._arc_prompt_tokens[request_id] = int(num_prompt_tokens)
    SingleTypeKVCacheManager.remove_skipped_blocks(
        self, request_id, processed_computed_tokens, num_prompt_tokens)
    last_state_block_idx = self.last_state_block_idx.get(request_id)
    if (
        last_state_block_idx is not None
        and last_state_block_idx < cdiv(processed_computed_tokens, self.block_size) - 1
    ):
        blocks = self.req_to_blocks[request_id]
        if blocks[last_state_block_idx] != self._null_block:
            freed = _arc_divert_hashed(
                self, request_id, [blocks[last_state_block_idx]], [last_state_block_idx]
            )
            if freed:
                self.block_pool.free_blocks(freed)
            blocks[last_state_block_idx] = self._null_block


def _arc_pop_blocks_for_free(self, request_id):
    if self.mamba_cache_mode == "align":
        # Capture any unretired hashed prompt-boundary AND decode-boundary blocks
        # still in req_to_blocks when a turn finishes (before remove_skipped_blocks
        # retired the final full prompt or decode block).
        req_blocks = self.req_to_blocks.get(request_id, ())
        prompt_tokens = self._arc_prompt_tokens.pop(request_id, None)
        cur_prompt_tok = _arc_max_bucket_tokens(self, request_id, decode_bucket=False)
        cur_decode_tok = _arc_max_bucket_tokens(self, request_id, decode_bucket=True)
        pinned_now = list(self._arc_pinned.get(request_id, ()))

        best_prompt_blk = None
        best_prompt_tok = cur_prompt_tok
        best_decode_cands = []
        for idx, blk in enumerate(req_blocks):
            if blk is None or blk.is_null or blk.block_hash is None:
                continue
            if any(b is blk for b in pinned_now):
                continue
            tok = _arc_block_tokens(self, blk, idx)
            if _arc_is_decode_tok(prompt_tokens, tok):
                if _ARC_PIN_DECODE and _ARC_PIN_DECODE_MAX > 0 and tok > cur_decode_tok:
                    best_decode_cands.append((tok, blk))
            else:
                if best_prompt_blk is None or tok >= best_prompt_tok:
                    best_prompt_blk = blk
                    best_prompt_tok = tok

        has_prompt_now = bool(self._arc_prompt_pins.get(request_id))
        if best_prompt_blk is not None and (not has_prompt_now or best_prompt_tok > cur_prompt_tok):
            best_prompt_blk.ref_cnt += 1
            _ARC_PIN_STATS["tail_pins"] += 1
            _arc_pin_add(self, request_id, best_prompt_blk, best_prompt_tok, is_decode=False)

        if best_decode_cands:
            best_decode_cands.sort(key=lambda item: item[0])
            for d_tok, d_blk in best_decode_cands[-max(1, _ARC_PIN_DECODE_MAX):]:
                d_blk.ref_cnt += 1
                _ARC_PIN_STATS["tail_pins"] += 1
                _ARC_PIN_STATS["decode_tail_pins"] += 1
                _arc_pin_add(self, request_id, d_blk, d_tok, is_decode=True)

    pin_tok_map = self._arc_pin_tokens.pop(request_id, {})
    self._arc_prompt_pins.pop(request_id, None)
    self._arc_decode_pins.pop(request_id, None)
    pins = self._arc_pinned.pop(request_id, None)
    res = _arc_orig_pop_blocks_for_free(self, request_id)
    if pins:
        _ARC_PIN_STATS["pinned_now"] -= len(pins)
        _ARC_PIN_STATS["request_released"] += len(pins)
        if _arc_handoff_supported(self):
            # Order older -> newer so only the newest _ARC_HANDOFF_PER_REQ (1) valid hashed snapshot
            # is held in _arc_handoff across the ~50 ms tool gap; older superseded pins return to MRU free_queue.
            pins.sort(key=lambda b: pin_tok_map.get(id(b), 0))
            valid_pins = [blk for blk in pins if not blk.is_null and blk.block_hash is not None]
            invalid_pins = [blk for blk in pins if blk.is_null or blk.block_hash is None]
            to_hold = valid_pins[-_ARC_HANDOFF_PER_REQ:] if valid_pins else []
            superseded = valid_pins[:-_ARC_HANDOFF_PER_REQ] if len(valid_pins) > _ARC_HANDOFF_PER_REQ else []
            immediate_free = invalid_pins + superseded
            if immediate_free:
                _ARC_PIN_STATS["handoff_superseded_released"] += len(superseded)
                self.block_pool.free_blocks(immediate_free)
            now = _arc_time.monotonic()
            for blk in to_hold:
                self._arc_handoff.append((blk, now))
                _ARC_PIN_STATS["handoff_held"] += 1
            # Sweep TTL and capacity cap only (min_free=0) because KVCacheManager.free
            # returns the finishing request's own blocks on the very next line.
            _arc_sweep_handoff(self, min_free=0)
        else:
            self.block_pool.free_blocks(pins)
    return res


def _arc_add_local_computed_blocks(self, request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens):
    res = _arc_orig_add_local_computed(
        self, request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens
    )
    if getattr(self, "_arc_handoff", None) and new_computed_blocks:
        hit_ids = {id(b) for b in new_computed_blocks if b is not None and not b.is_null}
        reused = []
        kept = []
        for blk, ts in self._arc_handoff:
            if id(blk) in hit_ids:
                reused.append(blk)
            else:
                kept.append((blk, ts))
        if reused:
            self._arc_handoff = kept
            _ARC_PIN_STATS["handoff_reused"] += len(reused)
            _ARC_PIN_STATS["handoff_now"] = sum(len(getattr(m, "_arc_handoff", ())) for m in _ARC_MANAGERS)
            self.block_pool.free_blocks(reused)
            if _ARC_PIN_LOG_EVERY > 0 and (
                _ARC_PIN_STATS["handoff_reused"] <= 3
                or _ARC_PIN_STATS["handoff_reused"] % 15 == 0
            ):
                logger.info("ARC_MAMBA_PIN handoff_reused %s", _ARC_PIN_STATS)
    return res


def _arc_get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks, *args, **kwargs):
    needed = _arc_orig_get_num_blocks_to_allocate(
        self, request_id, num_tokens, new_computed_blocks, *args, **kwargs
    )
    pool_cap = int(getattr(self.block_pool, "num_gpu_blocks", 1_000_000) or 1_000_000)
    if 0 <= int(needed) <= pool_cap:
        mgrs = _arc_pool_managers(self)
        self._arc_pending_keep[request_id] = tuple(new_computed_blocks or ())
        self._arc_pending_need[request_id] = int(needed)
        if self is mgrs[-1]:
            all_keep = []
            mamba_need = 0
            for m in mgrs:
                all_keep.extend(m._arc_pending_keep.pop(request_id, ()))
                mamba_need += int(m._arc_pending_need.pop(request_id, int(needed)))
            group_demand = mamba_need + max(12, int(needed) * 2)
            if _arc_pool_under_dual_pressure(self, extra_needed=group_demand):
                _arc_reclaim_pool_pins(self, _ARC_DUAL_PIN_MIN_FREE + group_demand)
            if any(getattr(m, "_arc_handoff", None) for m in mgrs):
                _arc_sweep_handoff(
                    self,
                    keep_blocks=all_keep,
                    min_free=max(_ARC_HANDOFF_MIN_FREE, group_demand),
                )
    return needed


if _ARC_PIN_ENABLED:
    MambaManager.__init__ = _arc_init
    MambaManager.allocate_new_blocks = _arc_allocate_new_blocks
    MambaManager._remove_blocks_in_range = _arc_remove_blocks_in_range
    MambaManager.remove_skipped_blocks = _arc_remove_skipped_blocks
    MambaManager.pop_blocks_for_free = _arc_pop_blocks_for_free
    if _arc_orig_cache_blocks is not None:
        MambaManager.cache_blocks = _arc_cache_blocks
    if _arc_orig_add_local_computed is not None:
        MambaManager.add_local_computed_blocks = _arc_add_local_computed_blocks
    if _arc_orig_get_num_blocks_to_allocate is not None:
        MambaManager.get_num_blocks_to_allocate = _arc_get_num_blocks_to_allocate
'''


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


TRITON_HELPERS_MARKER = "# ---- ARC_TRITON_ATTN_PATCH v2 ----"
TRITON_HELPERS_REPLACEMENTS = [
    (
        """@triton.jit
def resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    \"\"\"Resolve the (sequence, q-block-within-sequence) pair and load the
    per-sequence lengths.

    Shared across every attention kernel — the ``q_block_global_idx``
    program id indexes into the flattened ``(seq, q_block_in_seq)``
    space, and a binary search over ``query_start_len_ptr`` recovers
    the (seq, local-q-block) pair.

    Returns ``(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
    cur_batch_query_len, seq_len)``.  Callers must still early-return
    when ``q_block_local_idx * BLOCK_Q >= cur_batch_query_len`` (Triton
    helpers cannot return from the caller).
    \"\"\"
    # find_seq_idx is defined below; forward use is fine inside @triton.jit.
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx""",
        """@triton.jit
def resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    SINGLE_Q_BLOCK: tl.constexpr = False,
):
    \"\"\"Resolve the (sequence, q-block-within-sequence) pair and load the
    per-sequence lengths.

    Shared across every attention kernel — the ``q_block_global_idx``
    program id indexes into the flattened ``(seq, q_block_in_seq)``
    space, and a binary search over ``query_start_len_ptr`` recovers
    the (seq, local-q-block) pair.

    Returns ``(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
    cur_batch_query_len, seq_len)``.  Callers must still early-return
    when ``q_block_local_idx * BLOCK_Q >= cur_batch_query_len`` (Triton
    helpers cannot return from the caller).
    \"\"\"
    # find_seq_idx is defined below; forward use is fine inside @triton.jit.
    if SINGLE_Q_BLOCK:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
        )
        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
        q_block_local_idx = q_block_global_idx - q_block_start_idx""",
    ),
]

TRITON_UNIFIED_REPLACEMENTS = [
    (
        """    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)""",
        """    if KV_QUANT_MODE == 1:
        return data.to(Q.dtype)
    return data.to(Q.dtype)""",
    ),
    (
        """    MM_PREFIX_CLAMP_SW: tl.constexpr = False,
):
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = (KV_QUANT_MODE >= 2) and (
        KV_QUANT_MODE <= 3
    )
    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )""",
        """    MM_PREFIX_CLAMP_SW: tl.constexpr = False,
    SINGLE_Q_BLOCK: tl.constexpr = False,
):
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = (KV_QUANT_MODE >= 2) and (
        KV_QUANT_MODE <= 3
    )
    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr,
        seq_lens_ptr,
        q_block_global_idx,
        num_seqs,
        BLOCK_Q,
        SINGLE_Q_BLOCK,
    )""",
    ),
    (
        """    if USE_FP8_Q_DESCALE:
        score_scale = scale * tl.load(q_scale) * tl.load(k_scale)
        value_scale = tl.load(v_scale)""",
        """    if USE_FP8_Q_DESCALE:
        score_scale = (
            scale * tl.load(q_scale) * tl.load(k_scale)
            if Q_IS_FP8
            else scale * tl.load(k_scale)
        )
        value_scale = tl.load(v_scale)""",
    ),
    (
        """        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )""",
        """        segm_has_valid = (L > 0.0) & (loop_lo < loop_hi)
        M_segm = tl.where(segm_has_valid, M, float("-inf"))
        L_segm = tl.where(segm_has_valid, L, 0.0)
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M_segm,
            L_segm,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )""",
    ),
    (
        """    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums""",
        """    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)
    overall_max = tl.where(overall_max > float("-inf"), overall_max, 0.0)

    # load and rescale segment exp sums""",
    ),
    (
        """    tuned_large_head = (
        head_size == 256
        and max_seqlen_q > 1
        and num_queries_per_kv <= 16
        and current_platform.is_device_capability_family(100)
    )
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2

    # Ideally we would launch with kernel with:
    # \\sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \\sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \\sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \\sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    # Wider KV tile for the tuned large-head path (see above). Only the 2D
    # path (used when max_seqlen_q > 1) reads TILE_SIZE_PREFILL.
    if tuned_large_head:
        TILE_SIZE_PREFILL = 128""",
        """    tuned_large_head = (
        head_size == 256
        and max_seqlen_q > 1
        and num_queries_per_kv <= 16
    )
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2

    single_q_block = max_seqlen_q <= BLOCK_Q
    total_num_q_blocks = (
        num_seqs if single_q_block else (q.shape[0] // BLOCK_Q + num_seqs)
    )

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    # Wider KV tile for the tuned large-head path on Blackwell (SM100).
    if tuned_large_head and current_platform.is_device_capability_family(100):
        TILE_SIZE_PREFILL = 128""",
    ),
    (
        """    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )""",
        """    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or not single_q_block
        or q.shape[0] > softmax_segm_output.shape[0]
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )
    if tuned_large_head and use_3d:
        launch_num_warps = 4""",
    ),
    (
        """        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,
        **launch_kwargs,""",
        """        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,
        SINGLE_Q_BLOCK=single_q_block,
        **launch_kwargs,""",
    ),
    (
        """    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )""",
        """    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    UNIFORM_Q_LEN: tl.constexpr = False,
    MAX_SEQLEN_Q: tl.constexpr = 1,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if UNIFORM_Q_LEN:
        seq_idx = query_token_idx // MAX_SEQLEN_Q
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
        )""",
    ),
    (
        """            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )""",
        """            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
            UNIFORM_Q_LEN=(max_seqlen_q > 0 and q.shape[0] == num_seqs * max_seqlen_q),
            MAX_SEQLEN_Q=max(1, max_seqlen_q),
        )""",
    ),
]

TRITON_ATTN_REPLACEMENTS = [
    (
        """MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments""",
        """MIN_LAUNCH_GRID_SIZE_2D = 256  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 32  # Number of parallel tiled softmax segments""",
    ),
    (
        """        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )""",
        """        headdim_padded = next_power_of_2(self.headdim)
        max_segm_tokens = max(self.seq_threshold_3D * 5, 384)
        self.softmax_segm_output = torch.empty(
            (
                max_segm_tokens,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (max_segm_tokens, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (max_segm_tokens, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )""",
    ),
]

SMALL_TOPK_MARKER = "# ---- ARC_SMALL_TOPK_PATCH v4 ----"

ARC_SMALL_TOPK_MODULE_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
# ARC_SMALL_TOPK_PATCH v4: TP-sharded compact candidates + sort-free BF16 Top-K/Top-P + MTP draft/verify
from __future__ import annotations

import os
from typing import Any

import torch

try:
    from vllm.distributed import tensor_model_parallel_all_gather
except Exception:  # pragma: no cover
    tensor_model_parallel_all_gather = None

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # pragma: no cover
    logger = None

from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    tl_rand32,
    tl_rand64,
)

ARC_SMALL_K = 64
ARC_SMALL_TOPK_ENABLED = os.environ.get("ARC_SMALL_TOPK", "1").strip() not in (
    "0",
    "false",
    "False",
)
ARC_TP_LOCAL_TOPK_ENABLED = os.environ.get("ARC_TP_LOCAL_TOPK", "1").strip() not in (
    "0",
    "false",
    "False",
)
ARC_DRAFT_FP8_LM_HEAD_ENABLED = os.environ.get(
    "ARC_DRAFT_FP8_LM_HEAD", "1"
).strip() not in (
    "0",
    "false",
    "False",
)
ARC_DRAFT_MARLIN_LM_HEAD_ENABLED = os.environ.get(
    "ARC_DRAFT_MARLIN_LM_HEAD", "1"
).strip() not in (
    "0",
    "false",
    "False",
)

_ARC_SAMPLING_STATES_BY_DEVICE: dict[tuple[str, int], Any] = {}


def _norm_device_key(device: torch.device | str) -> tuple[str, int]:
    dev = torch.device(device)
    idx = dev.index
    if idx is None and dev.type == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
    return (dev.type, -1 if idx is None else int(idx))


def register_sampling_states(device: torch.device | str, states: Any) -> None:
    key = _norm_device_key(device)
    _ARC_SAMPLING_STATES_BY_DEVICE[key] = states
    if logger is not None:
        logger.info(
            "ARC_SMALL_TOPK_PATCH v4 registered sampling states (device=%s, K=%d, enabled=%s, tp_local=%s)",
            key,
            ARC_SMALL_K,
            ARC_SMALL_TOPK_ENABLED,
            ARC_TP_LOCAL_TOPK_ENABLED,
        )


def get_sampling_states(device: torch.device | str) -> Any | None:
    key = _norm_device_key(device)
    st = _ARC_SAMPLING_STATES_BY_DEVICE.get(key)
    if st is not None:
        return st
    if len(_ARC_SAMPLING_STATES_BY_DEVICE) == 1:
        return next(iter(_ARC_SAMPLING_STATES_BY_DEVICE.values()))
    return None


def _arc_select_sorted_topk(
    vals_f32: torch.Tensor,
    global_ids_i64: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean_vals = torch.where(torch.isnan(vals_f32), float("-inf"), vals_f32) + 0.0
    bits = clean_vals.view(torch.int32)
    ord_val = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    # Strict total order matching vLLM topk_topp_triton: (-logit, +global_token_id)
    sort_key = (ord_val.to(torch.int64) << 24) - global_ids_i64
    _, sel = torch.topk(sort_key, k, dim=-1, largest=True, sorted=True)
    return clean_vals.gather(-1, sel), global_ids_i64.gather(-1, sel)


def _resolve_lm_head_owner(model: Any) -> tuple[Any, Any, Any] | None:
    for obj in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        if obj is None:
            continue
        lm_head = getattr(obj, "lm_head", None)
        lp = getattr(obj, "logits_processor", None)
        if (
            lm_head is not None
            and lp is not None
            and hasattr(lm_head, "shard_indices")
            and not getattr(lp, "logits_as_input", False)
        ):
            shard_indices = lm_head.shard_indices
            if int(getattr(shard_indices, "num_added_elements_padded", 0)) != 0:
                return None
            org_vocab_size = int(getattr(lp, "org_vocab_size", 0))
            if org_vocab_size <= 0 or org_vocab_size >= (1 << 24):
                return None
            scale = float(getattr(lp, "scale", 1.0))
            if scale <= 0.0:
                return None
            soft_cap = getattr(lp, "soft_cap", None)
            if soft_cap is not None and float(soft_cap) <= 0.0:
                return None
            return obj, lm_head, lp
    return None


def resolve_lm_head_and_logits_processor(model: Any) -> tuple[Any, Any] | None:
    res = _resolve_lm_head_owner(model)
    if res is None:
        return None
    return res[1], res[2]


class _ArcFp8DraftQuantMethod:
    """Thin quant_method adapter delegating LogitsProcessor._apply_head to vLLM's FP8 linear kernel."""

    def __init__(self, fp8_linear: Any, out_features: int, out_dtype: torch.dtype) -> None:
        self.fp8_linear = fp8_linear
        self.out_features = int(out_features)
        self.out_dtype = out_dtype

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return x.new_empty((*x.shape[:-1], self.out_features), dtype=self.out_dtype)
            if not x.is_contiguous():
                x = x.contiguous()
        return self.fp8_linear.apply_weights(layer, x, bias)


class _ArcFp8DraftLMHead(torch.nn.Module):
    """Target-preserving draft-only W8A16-Marlin / W8A8-FP8 per-channel lm_head wrapper."""

    def __init__(
        self,
        base_head: Any,
        weight_kn: torch.Tensor,
        weight_scale: torch.Tensor,
        orig_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        k_cols = int(weight_kn.shape[0])
        n_rows = int(weight_kn.shape[1])
        for attr in (
            "tp_size",
            "tp_rank",
            "shard_indices",
            "org_vocab_size",
            "org_vocab_size_padded",
            "num_embeddings",
            "num_embeddings_padded",
            "num_embeddings_per_partition",
            "embedding_dim",
        ):
            if hasattr(base_head, attr):
                setattr(self, attr, getattr(base_head, attr))
        self.input_size_per_partition = k_cols
        self.output_size_per_partition = n_rows
        self.params_dtype = orig_dtype
        self.orig_dtype = orig_dtype
        self.logical_widths = [n_rows]
        self.weight_block_size = None
        self.bias = None
        self.input_scale = None
        self.input_scale_ub = None

        w_param = torch.nn.Parameter(weight_kn.data, requires_grad=False)
        w_param.input_dim = 0  # type: ignore[attr-defined]
        w_param.output_dim = 1  # type: ignore[attr-defined]
        self.weight = w_param
        self.weight_scale = torch.nn.Parameter(weight_scale.data, requires_grad=False)
        self.quant_method: Any = None


def _arc_quantize_fp8_channelwise_chunks(
    src_weight: torch.Tensor,
    fp8_dtype: torch.dtype,
    fp8_max: float = 448.0,
    chunk_rows: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `[N, K]` BF16/FP16 weight into per-output-channel FP8 `[N, K]` + FP32 `[N, 1]` scale in chunks."""
    n_rows, k_cols = int(src_weight.shape[0]), int(src_weight.shape[1])
    w_fp8_nk = torch.empty((n_rows, k_cols), dtype=fp8_dtype, device=src_weight.device)
    w_scale = torch.empty((n_rows, 1), dtype=torch.float32, device=src_weight.device)
    with torch.no_grad():
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            chunk = src_weight.data[start:end]
            amax = chunk.abs().amax(dim=1, keepdim=True).to(torch.float32).clamp_min(1e-12)
            scale = amax / fp8_max
            q = (chunk.to(torch.float32) / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
            w_fp8_nk[start:end].copy_(q)
            w_scale[start:end].copy_(scale)
            del chunk, amax, scale, q
    return w_fp8_nk, w_scale


def _arc_tp_min_flag(tp_cpu_group: Any, local_ok: bool) -> bool:
    if tp_cpu_group is None:
        return bool(local_ok)
    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=tp_cpu_group)
    return bool(int(flag.item()) == 1)


def maybe_install_arc_fp8_draft_lm_head(
    draft_model: Any,
    target_model: Any,
    vllm_config: Any = None,
    *,
    kernel_factory: Any = None,
    min_cc: int = 89,
) -> bool:
    """Install a separate W8A16-Marlin (preferred) or W8A8-FP8 per-channel lm_head on `draft_model` while leaving `target_model.lm_head` untouched."""
    if not ARC_DRAFT_FP8_LM_HEAD_ENABLED:
        return False
    try:
        draft_res = _resolve_lm_head_owner(draft_model)
        target_res = _resolve_lm_head_owner(target_model)
        if draft_res is None or target_res is None:
            return False
        draft_owner, draft_head, draft_lp = draft_res
        target_owner, target_head, _target_lp = target_res
        if draft_owner is target_owner:
            return False
        if isinstance(draft_head, _ArcFp8DraftLMHead):
            return True

        src_w = getattr(draft_head, "weight", None)
        tgt_w = getattr(target_head, "weight", None)
        if not isinstance(src_w, torch.Tensor) or not isinstance(tgt_w, torch.Tensor):
            return False
        if src_w.ndim != 2 or src_w.dtype not in (torch.bfloat16, torch.float16):
            return False
        if getattr(draft_head, "bias", None) is not None:
            return False
        lp_head_dtype = getattr(draft_lp, "head_dtype", None)
        if lp_head_dtype is not None and lp_head_dtype != src_w.dtype:
            return False

        n_rows, k_cols = int(src_w.shape[0]), int(src_w.shape[1])
        if n_rows < ARC_SMALL_K or (n_rows % 16) != 0 or (k_cols % 16) != 0:
            return False

        tp_cpu_group = None
        marlin_kernel_cls: Any = None
        free_b = 1 << 60
        if kernel_factory is None:
            if not src_w.is_cuda:
                return False
            cc_major, cc_minor = torch.cuda.get_device_capability(src_w.device)
            if (cc_major * 10 + cc_minor) < min_cc:
                return False
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    from vllm.distributed.parallel_state import get_tp_group

                    tp_grp = get_tp_group()
                    if int(getattr(tp_grp, "world_size", 1)) > 1:
                        tp_cpu_group = getattr(tp_grp, "cpu_group", None)
            except Exception:
                tp_cpu_group = None

            free_b, _total_b = torch.cuda.mem_get_info(src_w.device)
            # Require at least weight FP8 bytes (~303 MiB) + 128 MiB temporary cushion
            has_w8a8_vram = free_b >= (n_rows * k_cols + 128 * 1024 * 1024)
            if not _arc_tp_min_flag(tp_cpu_group, has_w8a8_vram):
                if logger is not None:
                    logger.warning(
                        "ARC_DRAFT_FP8_LM_HEAD skipped: insufficient free VRAM (%d MiB)",
                        free_b // (1024 * 1024),
                    )
                return False

            from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                process_fp8_weight_channel_strategy,
            )
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                kFp8DynamicTokenSym,
                kFp8StaticChannelSym,
            )
            from vllm.platforms import current_platform

            if ARC_DRAFT_MARLIN_LM_HEAD_ENABLED:
                try:
                    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
                        MarlinFP8ScaledMMLinearKernel,
                    )

                    marlin_kernel_cls = MarlinFP8ScaledMMLinearKernel
                except Exception:
                    marlin_kernel_cls = None

            fp8_dtype = current_platform.fp8_dtype()
            fp8_max = float(torch.finfo(fp8_dtype).max)
        else:
            factory_out = kernel_factory()
            if len(factory_out) == 7:
                (
                    fp8_dtype,
                    fp8_max,
                    process_fp8_weight_channel_strategy,
                    init_fp8_linear_kernel,
                    kFp8DynamicTokenSym,
                    kFp8StaticChannelSym,
                    marlin_kernel_cls,
                ) = factory_out
            else:
                (
                    fp8_dtype,
                    fp8_max,
                    process_fp8_weight_channel_strategy,
                    init_fp8_linear_kernel,
                    kFp8DynamicTokenSym,
                    kFp8StaticChannelSym,
                ) = factory_out
            if not ARC_DRAFT_MARLIN_LM_HEAD_ENABLED:
                marlin_kernel_cls = None

        orig_target_dtype = tgt_w.dtype
        target_ptr_before = int(tgt_w.data_ptr())
        shared_with_target = bool(
            (draft_head is target_head) or (int(src_w.data_ptr()) == target_ptr_before)
        )

        def _build_and_verify_candidate(force_cls: Any) -> tuple[_ArcFp8DraftLMHead, Any, float]:
            w_fp8_nk, w_scale = _arc_quantize_fp8_channelwise_chunks(
                src_w, fp8_dtype=fp8_dtype, fp8_max=fp8_max, chunk_rows=4096
            )
            weight_proc, scale_proc, _ = process_fp8_weight_channel_strategy(
                w_fp8_nk, w_scale, None
            )
            del w_fp8_nk, w_scale
            weight_kn = weight_proc.t()

            cand_head = _ArcFp8DraftLMHead(
                base_head=draft_head,
                weight_kn=weight_kn,
                weight_scale=scale_proc,
                orig_dtype=src_w.dtype,
            )
            init_kwargs: dict[str, Any] = {
                "activation_quant_key": kFp8DynamicTokenSym,
                "weight_quant_key": kFp8StaticChannelSym,
                "input_dtype": src_w.dtype,
                "out_dtype": src_w.dtype,
                "weight_shape": (n_rows, k_cols),
                "module_name": "ArcFp8DraftLMHead",
            }
            if force_cls is not None:
                init_kwargs["force_kernel"] = force_cls
            cand_linear = init_fp8_linear_kernel(**init_kwargs)
            cand_linear.process_weights_after_loading(cand_head)
            cand_head.quant_method = _ArcFp8DraftQuantMethod(
                cand_linear, out_features=n_rows, out_dtype=src_w.dtype
            )

            # Deterministic M=2 and M=24 numerical verification (zero global RNG side effects!)
            min_cos = 1.0
            with torch.no_grad():
                for probe_m in (2, 24):
                    probe_grid = (
                        torch.arange(
                            probe_m * k_cols, device=src_w.device, dtype=torch.float32
                        ).reshape(probe_m, k_cols)
                        * 0.0137
                    )
                    x_probe = torch.cos(probe_grid).to(src_w.dtype).contiguous()
                    ref_out = torch.nn.functional.linear(x_probe, src_w).to(torch.float32)
                    fp8_out = cand_head.quant_method.apply(
                        cand_head, x_probe, bias=None
                    ).to(torch.float32)
                    if tuple(fp8_out.shape) != (probe_m, n_rows) or not bool(
                        torch.isfinite(fp8_out).all().item()
                    ):
                        raise RuntimeError(
                            f"invalid FP8 draft lm_head output shape={tuple(fp8_out.shape)} (M={probe_m})"
                        )
                    ref_flat = ref_out.reshape(-1)
                    fp8_flat = fp8_out.reshape(-1)
                    denom = (
                        torch.linalg.vector_norm(ref_flat)
                        * torch.linalg.vector_norm(fp8_flat)
                    ).clamp_min(1e-12)
                    cos_m = float((torch.dot(ref_flat, fp8_flat) / denom).item())
                    if cos_m < 0.998:
                        raise RuntimeError(
                            f"FP8 draft lm_head cosine similarity too low (M={probe_m}): {cos_m:.6f}"
                        )
                    min_cos = min(min_cos, cos_m)
                    del probe_grid, x_probe, ref_out, fp8_out, ref_flat, fp8_flat
            return cand_head, cand_linear, min_cos

        fp8_head: _ArcFp8DraftLMHead | None = None
        fp8_linear: Any = None
        cos_sim = 0.0

        # 1) Try W8A16 MarlinFP8ScaledMMLinearKernel first if enabled & repack peak VRAM (~1.16 GiB) fits on all TP ranks
        can_try_marlin = bool(
            ARC_DRAFT_MARLIN_LM_HEAD_ENABLED
            and marlin_kernel_cls is not None
            and (n_rows % 64) == 0
            and (k_cols % 128) == 0
            and free_b >= (3 * n_rows * k_cols + 256 * 1024 * 1024)
        )
        if _arc_tp_min_flag(tp_cpu_group, can_try_marlin):
            marlin_cand: _ArcFp8DraftLMHead | None = None
            marlin_lin: Any = None
            marlin_cos = 0.0
            marlin_err: Exception | None = None
            try:
                marlin_cand, marlin_lin, marlin_cos = _build_and_verify_candidate(
                    marlin_kernel_cls
                )
            except Exception as exc:
                marlin_err = exc
                marlin_cand = None
                marlin_lin = None
                if src_w.is_cuda:
                    torch.cuda.empty_cache()
            if _arc_tp_min_flag(tp_cpu_group, marlin_cand is not None):
                fp8_head, fp8_linear, cos_sim = marlin_cand, marlin_lin, marlin_cos
            else:
                del marlin_cand, marlin_lin
                if src_w.is_cuda:
                    torch.cuda.empty_cache()
                if logger is not None and marlin_err is not None:
                    logger.warning(
                        "ARC_DRAFT_MARLIN_LM_HEAD fallback to W8A8 FP8 (%s: %s)",
                        type(marlin_err).__name__,
                        marlin_err,
                    )

        # 2) Fallback: build a completely fresh W8A8 FP8 candidate from untouched src_w
        if fp8_head is None:
            w8a8_cand: _ArcFp8DraftLMHead | None = None
            w8a8_lin: Any = None
            w8a8_cos = 0.0
            w8a8_err: Exception | None = None
            try:
                w8a8_cand, w8a8_lin, w8a8_cos = _build_and_verify_candidate(None)
            except Exception as exc:
                w8a8_err = exc
                w8a8_cand = None
                w8a8_lin = None
                if src_w.is_cuda:
                    torch.cuda.empty_cache()
            if not _arc_tp_min_flag(tp_cpu_group, w8a8_cand is not None):
                del w8a8_cand, w8a8_lin
                if src_w.is_cuda:
                    torch.cuda.empty_cache()
                if w8a8_err is not None:
                    raise w8a8_err
                raise RuntimeError("peer TP rank failed W8A8 draft lm_head installation")
            fp8_head, fp8_linear, cos_sim = w8a8_cand, w8a8_lin, w8a8_cos

        assert fp8_head is not None and fp8_linear is not None

        # Verify target_model.lm_head was never mutated
        if (
            getattr(target_owner, "lm_head", None) is not target_head
            or int(target_head.weight.data_ptr()) != target_ptr_before
            or target_head.weight.dtype != orig_target_dtype
        ):
            raise RuntimeError("target_model.lm_head invariant check failed")

        setattr(draft_owner, "lm_head", fp8_head)
        if not shared_with_target:
            del draft_head
        if src_w.is_cuda:
            torch.cuda.empty_cache()

        if logger is not None:
            logger.info(
                "ARC_DRAFT_FP8_LM_HEAD enabled (shape=%dx%d, kernel=%s, min_cos_sim=%.6f, shared_with_target=%s, target_dtype=%s)",
                n_rows,
                k_cols,
                type(fp8_linear).__name__,
                cos_sim,
                shared_with_target,
                orig_target_dtype,
            )
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "ARC_DRAFT_FP8_LM_HEAD fallback to BF16 draft lm_head (%s: %s)",
                type(exc).__name__,
                exc,
            )
        return False


def arc_compute_topk_candidates(
    model: Any,
    hidden_states: torch.Tensor,
    k: int = ARC_SMALL_K,
    all_gather_fn: Any = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not (ARC_SMALL_TOPK_ENABLED and ARC_TP_LOCAL_TOPK_ENABLED):
        return None
    resolved = resolve_lm_head_and_logits_processor(model)
    if resolved is None:
        return None
    lm_head, lp = resolved
    tp_size = int(getattr(lm_head, "tp_size", 1))
    gather_impl = (
        all_gather_fn if all_gather_fn is not None else tensor_model_parallel_all_gather
    )
    if tp_size > 1 and (
        gather_impl is None or not getattr(lp, "use_all_gather", True)
    ):
        return None

    shard_indices = lm_head.shard_indices
    if int(shard_indices.num_org_elements) < k:
        return None

    local_logits = lp._apply_head(lm_head, hidden_states, None)
    if local_logits is None or local_logits.shape[-1] < k:
        return None
    num_rows = local_logits.shape[0]
    if num_rows == 0:
        return (
            torch.empty((0, k), dtype=torch.float32, device=local_logits.device),
            torch.empty((0, k), dtype=torch.int64, device=local_logits.device),
        )

    num_pad = int(shard_indices.num_org_vocab_padding)
    if num_pad > 0:
        local_logits[..., -num_pad:] = float("-inf")
    local_logits.nan_to_num_(
        nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")
    )

    local_vals, local_idx = torch.topk(
        local_logits, k, dim=-1, largest=True, sorted=False
    )
    if lp.soft_cap is not None:
        local_vals = torch.tanh(local_vals / lp.soft_cap) * lp.soft_cap
    if lp.scale != 1.0:
        local_vals = local_vals * lp.scale

    vals_f32 = local_vals.to(torch.float32)
    vocab_start = int(shard_indices.org_vocab_start_index)

    if tp_size == 1:
        global_ids_i64 = local_idx.to(torch.int64) + vocab_start
        return _arc_select_sorted_topk(vals_f32, global_ids_i64, k)

    global_ids_i32 = local_idx.to(torch.int32) + vocab_start
    packed = torch.cat([vals_f32.view(torch.int32), global_ids_i32], dim=-1)
    gathered = gather_impl(packed, dim=0)
    gathered_4d = gathered.view(tp_size, num_rows, 2, k)
    all_vals_f32 = (
        gathered_4d[:, :, 0, :]
        .permute(1, 0, 2)
        .reshape(num_rows, tp_size * k)
        .view(torch.float32)
    )
    all_ids_i64 = (
        gathered_4d[:, :, 1, :]
        .permute(1, 0, 2)
        .reshape(num_rows, tp_size * k)
        .to(torch.int64)
    )
    return _arc_select_sorted_topk(all_vals_f32, all_ids_i64, k)


@triton.jit
def _arc_small_k_trunc_probs(
    raw_vals,
    cols,
    temp,
    req_k,
    req_p,
    BLOCK_K: tl.constexpr,
):
    is_nan = raw_vals != raw_vals
    vals = tl.where(is_nan, float("-inf"), raw_vals)
    if temp > 0.0:
        if temp != 1.0:
            vals = vals / temp
        k_eff = tl.minimum(req_k, BLOCK_K)
        k_eff = tl.maximum(k_eff, 1)
        vals = tl.where(cols < k_eff, vals, float("-inf"))
        row_max = tl.max(vals, axis=0)
        has_finite = row_max > float("-inf")
        exps = tl.where(has_finite, tl.exp(vals - row_max), 0.0)
        sum_exps = tl.sum(exps, axis=0)
        probs = tl.where(
            sum_exps > 0.0,
            exps / sum_exps,
            tl.where(cols == 0, 1.0, 0.0),
        )
        if req_p < 1.0:
            csum = tl.cumsum(probs, axis=0)
            keep_p = (cols == 0) | ((csum - probs) < req_p)
            vals = tl.where(keep_p, vals, float("-inf"))
            probs = tl.where(keep_p, probs, 0.0)
            norm_p = tl.sum(probs, axis=0)
            probs = tl.where(
                norm_p > 0.0,
                probs / norm_p,
                tl.where(cols == 0, 1.0, 0.0),
            )
        all_inf = tl.max(vals, axis=0) == float("-inf")
        vals = tl.where(all_inf & (cols == 0), 0.0, vals)
    else:
        vals = tl.where(cols == 0, 0.0, float("-inf"))
        probs = tl.where(cols == 0, 1.0, 0.0)
    return vals, probs


@triton.jit
def _arc_small_k_gumbel_pick(
    vals,
    tok_ids,
    cols,
    seed,
    pos,
    temp,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    if temp == 0.0:
        best_col = 0
    else:
        draw_pos = pos + _DRAFT_NOISE_SALT if IS_DRAFTING else pos
        gumbel_seed = tl.randint(seed, draw_pos)
        tok_ids_i32 = tok_ids.to(tl.int32)
        valid = vals > float("-inf")
        if USE_FP64:
            u = tl_rand64(gumbel_seed, tok_ids_i32, includes_zero=False)
            g_noise = -tl.log(-tl.log(u))
            scores = tl.where(valid, vals.to(tl.float64) + g_noise, float("-inf"))
        else:
            u = tl_rand32(gumbel_seed, tok_ids_i32, includes_zero=False)
            g_noise = -tl.log(-tldevice.log1p(-u))
            scores = tl.where(valid, vals + g_noise, float("-inf"))
        scores = tl.where(scores != scores, float("-inf"), scores)
        best_col = tl.argmax(scores, axis=0)
    return tl.sum(tl.where(cols == best_col, tok_ids, 0), axis=0)


@triton.jit
def _arc_small_k_sample_kernel(
    sampled_ptr,
    vals_ptr,
    vals_stride,
    idx_ptr,
    idx_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_k_ptr,
    top_p_ptr,
    seeds_ptr,
    pos_ptr,
    BLOCK_K: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    is_valid = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid, other=0.0).to(tl.float32)
    req_k = tl.load(top_k_ptr + req_state_idx, mask=is_valid, other=BLOCK_K).to(
        tl.int32
    )
    req_p = tl.load(top_p_ptr + req_state_idx, mask=is_valid, other=1.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid, other=0)
    pos = tl.load(pos_ptr + row).to(tl.int64)

    raw_vals = tl.load(vals_ptr + row * vals_stride + cols).to(tl.float32)
    tok_ids = tl.load(idx_ptr + row * idx_stride + cols).to(tl.int64)

    vals, _ = _arc_small_k_trunc_probs(raw_vals, cols, temp, req_k, req_p, BLOCK_K)
    sampled_tok = _arc_small_k_gumbel_pick(
        vals,
        tok_ids,
        cols,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
    )
    tl.store(sampled_ptr + row, sampled_tok)


@triton.jit
def _arc_small_k_draft_sample_kernel(
    sampled_ptr,
    cache_idx_ptr,
    cache_idx_stride_0,
    cache_idx_stride_1,
    cache_probs_ptr,
    cache_probs_stride_0,
    cache_probs_stride_1,
    vals_ptr,
    vals_stride,
    idx_ptr,
    idx_stride,
    idx_mapping_ptr,
    temp_ptr,
    top_k_ptr,
    top_p_ptr,
    seeds_ptr,
    pos_ptr,
    draft_step_ptr,
    BLOCK_K: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    is_valid = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid, other=0.0).to(tl.float32)
    req_k = tl.load(top_k_ptr + req_state_idx, mask=is_valid, other=BLOCK_K).to(
        tl.int32
    )
    req_p = tl.load(top_p_ptr + req_state_idx, mask=is_valid, other=1.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid, other=0)
    pos = tl.load(pos_ptr + row).to(tl.int64)
    col_step = tl.load(draft_step_ptr).to(tl.int64)

    raw_vals = tl.load(vals_ptr + row * vals_stride + cols).to(tl.float32)
    tok_ids = tl.load(idx_ptr + row * idx_stride + cols).to(tl.int64)

    vals, probs = _arc_small_k_trunc_probs(raw_vals, cols, temp, req_k, req_p, BLOCK_K)

    tl.store(
        cache_idx_ptr
        + req_state_idx * cache_idx_stride_0
        + col_step * cache_idx_stride_1
        + cols,
        tok_ids.to(tl.int32),
        mask=is_valid,
    )
    tl.store(
        cache_probs_ptr
        + req_state_idx * cache_probs_stride_0
        + col_step * cache_probs_stride_1
        + cols,
        probs,
        mask=is_valid,
    )

    sampled_tok = _arc_small_k_gumbel_pick(
        vals,
        tok_ids,
        cols,
        seed,
        pos,
        temp,
        IS_DRAFTING=True,
        USE_FP64=USE_FP64,
    )
    tl.store(sampled_ptr + row, sampled_tok)


@triton.jit
def _arc_small_k_target_probs_kernel(
    out_probs_ptr,
    out_probs_stride,
    vals_ptr,
    vals_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_k_ptr,
    top_p_ptr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    is_valid = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid, other=0.0).to(tl.float32)
    req_k = tl.load(top_k_ptr + req_state_idx, mask=is_valid, other=BLOCK_K).to(
        tl.int32
    )
    req_p = tl.load(top_p_ptr + req_state_idx, mask=is_valid, other=1.0).to(tl.float32)
    raw_vals = tl.load(vals_ptr + row * vals_stride + cols).to(tl.float32)
    _, probs = _arc_small_k_trunc_probs(raw_vals, cols, temp, req_k, req_p, BLOCK_K)
    tl.store(out_probs_ptr + row * out_probs_stride + cols, probs)


@triton.jit
def _arc_small_k_rejection_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    target_idx_ptr,
    target_idx_stride,
    target_probs_ptr,
    target_probs_stride,
    draft_idx_ptr,
    draft_idx_stride_0,
    draft_idx_stride_1,
    draft_probs_ptr,
    draft_probs_stride_0,
    draft_probs_stride_1,
    draft_sampled_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    num_speculative_steps,
    BLOCK_K: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_draft_tokens = tl.maximum(end_idx - start_idx - 1, 0)
    seed = tl.load(seeds_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    for j in range(num_speculative_steps + 1):
        tl.store(sampled_ptr + req_idx * sampled_stride + j, -1)

    accepted_length = tl.zeros((), tl.int64)
    verifying = 1
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_tok = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        is_valid_draft = draft_tok >= 0
        if not is_greedy:
            verifying = verifying & tl.where(is_valid_draft, 1, 0)
        if verifying != 0:
            pos_i = tl.load(pos_ptr + logit_idx).to(tl.int64)
            u = tl_rand32(seed, pos_i, includes_zero=False)
            if is_greedy:
                target_argmax = tl.load(
                    target_idx_ptr + logit_idx * target_idx_stride
                ).to(tl.int64)
                acc = (target_argmax == draft_tok) & is_valid_draft
                verifying = tl.where(acc, 1, 0)
                accepted_length += tl.where(acc, 1, 0)
                out_tok = tl.where(acc, draft_tok, target_argmax)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, out_tok)
            else:
                t_ids = tl.load(
                    target_idx_ptr + logit_idx * target_idx_stride + cols
                ).to(tl.int64)
                t_probs = tl.load(
                    target_probs_ptr + logit_idx * target_probs_stride + cols
                ).to(tl.float32)
                p_x = tl.sum(tl.where(t_ids == draft_tok, t_probs, 0.0), axis=0)

                d_ids = tl.load(
                    draft_idx_ptr
                    + req_state_idx * draft_idx_stride_0
                    + i * draft_idx_stride_1
                    + cols
                ).to(tl.int64)
                d_probs = tl.load(
                    draft_probs_ptr
                    + req_state_idx * draft_probs_stride_0
                    + i * draft_probs_stride_1
                    + cols
                ).to(tl.float32)
                q_x = tl.sum(tl.where(d_ids == draft_tok, d_probs, 0.0), axis=0)

                acc = p_x > (u * q_x)
                verifying = tl.where(acc, 1, 0)
                accepted_length += tl.where(acc, 1, 0)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_tok)

    resample_idx = accepted_length
    resample_token_idx = start_idx + resample_idx
    is_bonus = resample_token_idx == (end_idx - 1)
    tl.store(num_sampled_ptr + req_idx, (resample_idx + 1).to(tl.int32))

    if is_greedy and not is_bonus:
        return

    if is_greedy:
        bonus_argmax = tl.load(
            target_idx_ptr + resample_token_idx * target_idx_stride
        ).to(tl.int64)
        tl.store(sampled_ptr + req_idx * sampled_stride + resample_idx, bonus_argmax)
        return

    t_ids = tl.load(
        target_idx_ptr + resample_token_idx * target_idx_stride + cols
    ).to(tl.int64)
    t_probs = tl.load(
        target_probs_ptr + resample_token_idx * target_probs_stride + cols
    ).to(tl.float32)

    rejected_draft_tok = tl.load(
        draft_sampled_ptr + resample_token_idx + 1,
        mask=not is_bonus,
        other=-1,
    ).to(tl.int64)

    if is_bonus or (rejected_draft_tok < 0):
        res_probs = t_probs
    else:
        d_ids = tl.load(
            draft_idx_ptr
            + req_state_idx * draft_idx_stride_0
            + resample_idx * draft_idx_stride_1
            + cols
        ).to(tl.int64)
        d_probs = tl.load(
            draft_probs_ptr
            + req_state_idx * draft_probs_stride_0
            + resample_idx * draft_probs_stride_1
            + cols
        ).to(tl.float32)
        match = t_ids[:, None] == d_ids[None, :]
        q_on_t = tl.sum(tl.where(match, d_probs[None, :], 0.0), axis=1)
        res_probs = tl.maximum(t_probs - q_on_t, 0.0)
        res_sum = tl.sum(res_probs, axis=0)
        res_probs = tl.where(res_sum > 0.0, res_probs, t_probs)

    pos_r = tl.load(pos_ptr + resample_token_idx).to(tl.int64)
    res_logits = tl.where(res_probs > 0.0, tl.log(res_probs), float("-inf"))
    resampled_tok = _arc_small_k_gumbel_pick(
        res_logits,
        t_ids,
        cols,
        seed,
        pos_r,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
    )
    tl.store(sampled_ptr + req_idx * sampled_stride + resample_idx, resampled_tok)


def arc_small_k_sample_candidates(
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    num_tokens = topk_vals.shape[0]
    sampled = torch.empty(num_tokens, dtype=torch.int64, device=topk_vals.device)
    if num_tokens == 0:
        return sampled
    _arc_small_k_sample_kernel[(num_tokens,)](
        sampled,
        topk_vals,
        topk_vals.stride(0),
        topk_idx,
        topk_idx.stride(0),
        expanded_idx_mapping.contiguous(),
        temperature,
        top_k,
        top_p,
        seeds,
        pos.contiguous(),
        BLOCK_K=ARC_SMALL_K,
        USE_FP64=use_fp64,
        num_warps=2,
    )
    return sampled


def arc_small_k_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    raw_vals, raw_idx = torch.topk(
        logits, ARC_SMALL_K, dim=-1, largest=True, sorted=False
    )
    topk_vals, topk_idx = _arc_select_sorted_topk(
        raw_vals.to(torch.float32), raw_idx, ARC_SMALL_K
    )
    return arc_small_k_sample_candidates(
        topk_vals,
        topk_idx,
        expanded_idx_mapping,
        temperature,
        top_k,
        top_p,
        seeds,
        pos,
        use_fp64=use_fp64,
    )


def arc_small_k_draft_sample_candidates(
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    draft_step: torch.Tensor,
    cache_idx: torch.Tensor,
    cache_probs: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    num_tokens = topk_vals.shape[0]
    sampled = torch.empty(num_tokens, dtype=torch.int64, device=topk_vals.device)
    if num_tokens == 0:
        return sampled
    _arc_small_k_draft_sample_kernel[(num_tokens,)](
        sampled,
        cache_idx,
        cache_idx.stride(0),
        cache_idx.stride(1),
        cache_probs,
        cache_probs.stride(0),
        cache_probs.stride(1),
        topk_vals,
        topk_vals.stride(0),
        topk_idx,
        topk_idx.stride(0),
        idx_mapping.contiguous(),
        temperature,
        top_k,
        top_p,
        seeds,
        pos.contiguous(),
        draft_step.contiguous(),
        BLOCK_K=ARC_SMALL_K,
        USE_FP64=use_fp64,
        num_warps=2,
    )
    return sampled


def arc_small_k_draft_sample(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    draft_step: torch.Tensor,
    cache_idx: torch.Tensor,
    cache_probs: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    raw_vals, raw_idx = torch.topk(
        logits, ARC_SMALL_K, dim=-1, largest=True, sorted=False
    )
    topk_vals, topk_idx = _arc_select_sorted_topk(
        raw_vals.to(torch.float32), raw_idx, ARC_SMALL_K
    )
    return arc_small_k_draft_sample_candidates(
        topk_vals,
        topk_idx,
        idx_mapping,
        temperature,
        top_k,
        top_p,
        seeds,
        pos,
        draft_step,
        cache_idx,
        cache_probs,
        use_fp64=use_fp64,
    )


def arc_small_k_verify_candidates(
    target_vals: torch.Tensor,
    target_idx: torch.Tensor,
    draft_cache_idx: torch.Tensor,
    draft_cache_probs: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seeds: torch.Tensor,
    num_speculative_steps: int,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_logits = target_vals.shape[0]
    num_reqs = idx_mapping.shape[0]
    target_probs = torch.empty(
        (num_logits, ARC_SMALL_K), dtype=torch.float32, device=target_vals.device
    )
    if num_logits > 0:
        _arc_small_k_target_probs_kernel[(num_logits,)](
            target_probs,
            target_probs.stride(0),
            target_vals,
            target_vals.stride(0),
            expanded_idx_mapping.contiguous(),
            temperature,
            top_k,
            top_p,
            BLOCK_K=ARC_SMALL_K,
            num_warps=2,
        )
    sampled = torch.empty(
        (num_reqs, num_speculative_steps + 1),
        dtype=torch.int64,
        device=target_vals.device,
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=target_vals.device)
    if num_reqs > 0:
        _arc_small_k_rejection_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            target_idx,
            target_idx.stride(0),
            target_probs,
            target_probs.stride(0),
            draft_cache_idx,
            draft_cache_idx.stride(0),
            draft_cache_idx.stride(1),
            draft_cache_probs,
            draft_cache_probs.stride(0),
            draft_cache_probs.stride(1),
            draft_sampled.contiguous(),
            cu_num_logits.contiguous(),
            idx_mapping.contiguous(),
            temperature,
            seeds,
            pos.contiguous(),
            num_speculative_steps,
            BLOCK_K=ARC_SMALL_K,
            USE_FP64=use_fp64,
            num_warps=2,
        )
    return sampled, num_sampled


def arc_small_k_verify(
    logits: torch.Tensor,
    draft_cache_idx: torch.Tensor,
    draft_cache_probs: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seeds: torch.Tensor,
    num_speculative_steps: int,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_vals, raw_idx = torch.topk(
        logits, ARC_SMALL_K, dim=-1, largest=True, sorted=False
    )
    target_vals, target_idx = _arc_select_sorted_topk(
        raw_vals.to(torch.float32), raw_idx, ARC_SMALL_K
    )
    return arc_small_k_verify_candidates(
        target_vals,
        target_idx,
        draft_cache_idx,
        draft_cache_probs,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        temperature,
        top_k,
        top_p,
        seeds,
        num_speculative_steps,
        use_fp64=use_fp64,
    )


def arc_small_k_reconstruct_full_draft_logits(
    draft_cache_idx: torch.Tensor,
    draft_cache_probs: torch.Tensor,
    temperature: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    max_reqs, num_steps, _ = draft_cache_idx.shape
    full = torch.full(
        (max_reqs, num_steps, vocab_size),
        float("-inf"),
        dtype=torch.float32,
        device=draft_cache_probs.device,
    )
    temps = temperature.clamp_min(1e-6).view(-1, 1, 1)
    src = torch.where(
        draft_cache_probs > 0.0,
        torch.log(draft_cache_probs.clamp_min(1e-38)) * temps,
        torch.full_like(draft_cache_probs, float("-inf")),
    )
    full.scatter_(-1, draft_cache_idx.to(torch.int64), src)
    return full
'''

SAMPLER_REPLACEMENTS = [
    (
        """from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.sample.trace_replay import TraceReplayState
from vllm.v1.worker.gpu.states import RequestState""",
        """from vllm.v1.worker.gpu.sample.arc_small_topk import (
    ARC_SMALL_K,
    ARC_SMALL_TOPK_ENABLED,
    arc_small_k_sample,
    arc_small_k_sample_candidates,
    register_sampling_states,
)
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.sample.trace_replay import TraceReplayState
from vllm.v1.worker.gpu.states import RequestState""",
    ),
    (
        """        self.needs_logits_processing = np.zeros(max_num_reqs, dtype=bool)
        self.num_speculative_tokens = num_speculative_tokens""",
        """        self.needs_logits_processing = np.zeros(max_num_reqs, dtype=bool)
        self.needs_heavy_logits_processing = np.zeros(max_num_reqs, dtype=bool)
        register_sampling_states(device, self.sampling_states)
        self.num_speculative_tokens = num_speculative_tokens""",
    ),
    (
        """        states = self.sampling_states
        temperature = states.temperature.np[req_idx]
        self.needs_logits_processing[req_idx] = (
            self.logit_bias_state.use_logit_bias[req_idx]
            or self.penalties_state.use_penalty[req_idx]
            or self.bad_words_state.num_bad_words.np[req_idx] > 0
            or (
                self.thinking_budget_state.enabled
                and self.thinking_budget_state.use_thinking_budget[req_idx]
            )
            or (temperature != 0.0 and temperature != 1.0)
            or states.min_p.np[req_idx] != 0.0
            or states.top_k.np[req_idx] != states.vocab_size
            or states.top_p.np[req_idx] != 1.0
        )""",
        """        states = self.sampling_states
        heavy = (
            self.logit_bias_state.use_logit_bias[req_idx]
            or self.penalties_state.use_penalty[req_idx]
            or self.bad_words_state.num_bad_words.np[req_idx] > 0
            or (
                self.thinking_budget_state.enabled
                and self.thinking_budget_state.use_thinking_budget[req_idx]
            )
            or states.min_p.np[req_idx] != 0.0
        )
        self.needs_heavy_logits_processing[req_idx] = heavy
        self.needs_logits_processing[req_idx] = (
            heavy
            or states.top_k.np[req_idx] != states.vocab_size
            or states.top_p.np[req_idx] != 1.0
        )

    def can_use_small_topk_fastpath(
        self, idx_mapping_np: np.ndarray, return_logprobs: bool = False
    ) -> bool:
        if not ARC_SMALL_TOPK_ENABLED or idx_mapping_np.size == 0:
            return False
        if self.return_sampling_mask:
            return False
        if return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            return False
        states = self.sampling_states
        if states.vocab_size < ARC_SMALL_K:
            return False
        if np.any(self.needs_heavy_logits_processing[idx_mapping_np]):
            return False
        top_k_np = states.top_k.np[idx_mapping_np]
        temp_np = states.temperature.np[idx_mapping_np]
        return bool(np.all((top_k_np <= ARC_SMALL_K) | (temp_np == 0.0)))

    def sample_from_candidates(
        self,
        topk_vals: torch.Tensor,
        topk_idx: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        pos = input_batch.positions[input_batch.logits_indices]
        sampled = arc_small_k_sample_candidates(
            topk_vals,
            topk_idx,
            input_batch.expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.top_k.gpu,
            self.sampling_states.top_p.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            use_fp64=self.use_fp64_gumbel,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=None,
        )""",
    ),
    (
        """    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed_logits = self.apply_sampling_params(""",
        """    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.can_use_small_topk_fastpath(
            idx_mapping_np, return_logprobs=return_logprobs
        ):
            sampled = arc_small_k_sample(
                logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.top_k.gpu,
                self.sampling_states.top_p.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                use_fp64=self.use_fp64_gumbel,
            )
            return sampled, logits
        processed_logits = self.apply_sampling_params(""",
    ),
    (
        """            sampled = gumbel_sample(
                processed_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                is_drafting=False,
                use_fp64=self.use_fp64_gumbel,
            )""",
        """            sampled = gumbel_sample(
                processed_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=(processed_logits.dtype != torch.float32),
                is_drafting=False,
                use_fp64=self.use_fp64_gumbel,
            )""",
    ),
]

SPECULATOR_REPLACEMENTS = [
    (
        """from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)""",
        """from vllm.v1.worker.gpu.sample.arc_small_topk import (
    ARC_SMALL_K,
    ARC_SMALL_TOPK_ENABLED,
    arc_compute_topk_candidates,
    arc_small_k_draft_sample,
    arc_small_k_draft_sample_candidates,
    get_sampling_states,
    maybe_install_arc_fp8_draft_lm_head,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)""",
    ),
    (
        """        self.model = self.load_draft_model(target_model, target_attn_layer_names)
        self._validate_local_argmax_reduction()""",
        """        self.model = self.load_draft_model(target_model, target_attn_layer_names)
        self._validate_local_argmax_reduction()
        if not self.use_local_argmax_reduction:
            maybe_install_arc_fp8_draft_lm_head(
                self.model, target_model, self.vllm_config
            )""",
    ),
    (
        """        self.draft_watermarker: DraftWatermarker | None = None
        watermark_config = getattr(vllm_config, "watermark_config", None)
        if watermark_config is not None:
            watermarker = create_watermarker(watermark_config)
            self.draft_watermarker = create_speculative_draft_watermarker(
                watermarker,
                self.max_num_reqs,
                device,
                watermark_config.allow_target_only_watermarking,
            )

        self.supports_mm_inputs = False""",
        """        self.draft_watermarker: DraftWatermarker | None = None
        watermark_config = getattr(vllm_config, "watermark_config", None)
        if watermark_config is not None:
            watermarker = create_watermarker(watermark_config)
            self.draft_watermarker = create_speculative_draft_watermarker(
                watermarker,
                self.max_num_reqs,
                device,
                watermark_config.allow_target_only_watermarking,
            )

        self.arc_top_k = torch.full(
            (self.max_num_reqs,), ARC_SMALL_K, dtype=torch.int32, device=device
        )
        self.arc_top_p = torch.ones(
            (self.max_num_reqs,), dtype=torch.float32, device=device
        )
        self.arc_draft_topk_idx: torch.Tensor | None = None
        self.arc_draft_topk_probs: torch.Tensor | None = None
        if (
            ARC_SMALL_TOPK_ENABLED
            and self.draft_logits is not None
            and self.draft_watermarker is None
            and not self.use_acceptance_estimator
            and self.vocab_size >= ARC_SMALL_K
        ):
            dtype, fill = self.draft_logits_spec(vllm_config)
            self.draft_logits = torch.full(
                (1, 1, 1), fill, dtype=dtype, device=device
            )
            self.arc_draft_topk_idx = torch.zeros(
                (self.max_num_reqs, self.num_speculative_steps, ARC_SMALL_K),
                dtype=torch.int32,
                device=device,
            )
            self.arc_draft_topk_probs = torch.zeros(
                (self.max_num_reqs, self.num_speculative_steps, ARC_SMALL_K),
                dtype=torch.float32,
                device=device,
            )
            self.draft_logits._arc_small_k_cache = (
                self.arc_draft_topk_idx,
                self.arc_draft_topk_probs,
            )
            logger.info(
                "ARC_SMALL_TOPK_PATCH v4 enabled for DraftModelSpeculator (K=%d, steps=%d, vocab=%d)",
                ARC_SMALL_K,
                self.num_speculative_steps,
                self.vocab_size,
            )

        self.supports_mm_inputs = False""",
    ),
    (
        """        if draft_logits is not None:
            logits = self.model.compute_logits(hidden_states)
            sampled = gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                sample_src_positions,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
            )""",
        """        if draft_logits is not None:
            small_k_cache = getattr(draft_logits, "_arc_small_k_cache", None)
            if small_k_cache is not None and self.acceptance_estimator is None:
                cand = arc_compute_topk_candidates(
                    self.model, hidden_states, ARC_SMALL_K
                )
                if cand is not None:
                    cand_vals, cand_idx = cand
                    return arc_small_k_draft_sample_candidates(
                        cand_vals,
                        cand_idx,
                        idx_mapping,
                        temperature,
                        self.arc_top_k,
                        self.arc_top_p,
                        seeds,
                        sample_src_positions,
                        draft_step,
                        small_k_cache[0],
                        small_k_cache[1],
                        use_fp64=self.use_fp64_gumbel,
                    )
            logits = self.model.compute_logits(hidden_states)
            if small_k_cache is not None:
                sampled = arc_small_k_draft_sample(
                    logits,
                    idx_mapping,
                    temperature,
                    self.arc_top_k,
                    self.arc_top_p,
                    seeds,
                    sample_src_positions,
                    draft_step,
                    small_k_cache[0],
                    small_k_cache[1],
                    use_fp64=self.use_fp64_gumbel,
                )
                self._maybe_predict_acceptance(logits, idx_mapping, draft_step)
                return sampled
            sampled = gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                sample_src_positions,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
            )""",
    ),
    (
        """        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self.idx_mapping[:num_reqs].copy_(idx_mapping)""",
        """        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        sampling_states = get_sampling_states(self.device)
        if sampling_states is not None:
            self.arc_top_k.copy_(sampling_states.top_k.gpu)
            self.arc_top_p.copy_(sampling_states.top_p.gpu)
        self.idx_mapping[:num_reqs].copy_(idx_mapping)""",
    ),
]

REJECTION_SAMPLER_REPLACEMENTS = [
    (
        """from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)""",
        """from vllm.v1.worker.gpu.sample.arc_small_topk import (
    arc_small_k_reconstruct_full_draft_logits,
    arc_small_k_verify,
    arc_small_k_verify_candidates,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)""",
    ),
    (
        """    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(""",
        """    def verify_from_candidates(
        self,
        target_vals: torch.Tensor,
        target_idx: torch.Tensor,
        input_batch: InputBatch,
        small_k_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> SamplerOutput:
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        sampled, num_sampled = arc_small_k_verify_candidates(
            target_vals,
            target_idx,
            small_k_cache[0],
            small_k_cache[1],
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.top_k.gpu,
            self.sampler.sampling_states.top_p.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            use_fp64=self.sampler.use_fp64_gumbel,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        small_k_cache = (
            getattr(draft_logits, "_arc_small_k_cache", None)
            if draft_logits is not None
            else None
        )
        if small_k_cache is not None:
            return_logprobs = (
                self.sampler.sampling_states.max_num_logprobs(idx_mapping_np)
                != NO_LOGPROBS
            )
            if (
                self.watermark_key is None
                and not self.use_block_verification
                and self.synthetic_conditional_rates is None
                and self.sampler.can_use_small_topk_fastpath(
                    idx_mapping_np, return_logprobs=return_logprobs
                )
            ):
                sampled, num_sampled = arc_small_k_verify(
                    logits,
                    small_k_cache[0],
                    small_k_cache[1],
                    draft_sampled,
                    cu_num_logits,
                    pos,
                    idx_mapping,
                    expanded_idx_mapping,
                    self.sampler.sampling_states.temperature.gpu,
                    self.sampler.sampling_states.top_k.gpu,
                    self.sampler.sampling_states.top_p.gpu,
                    self.sampler.sampling_states.seeds.gpu,
                    self.num_speculative_steps,
                    use_fp64=self.sampler.use_fp64_gumbel,
                )
                return logits, sampled, num_sampled
            draft_logits = arc_small_k_reconstruct_full_draft_logits(
                small_k_cache[0],
                small_k_cache[1],
                self.sampler.sampling_states.temperature.gpu,
                logits.shape[1],
            )
        processed_logits = self.sampler.apply_sampling_params(""",
    ),
]

MODEL_RUNNER_REPLACEMENTS = [
    (
        """from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler""",
        """from vllm.v1.worker.gpu.sample.arc_small_topk import (
    ARC_SMALL_K,
    arc_compute_topk_candidates,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler""",
    ),
    (
        """        else:
            sample_hidden_states = hidden_states[input_batch.logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)""",
        """        else:
            sample_hidden_states = hidden_states[input_batch.logits_indices]
            if (
                grammar_output is None
                and input_batch.num_reqs > 0
                and self.sampler is not None
                and not self.sampler.compute_nans
                and self.sampler.trace_replay_state is None
                and self.sampler.get_logprobs_dims(input_batch.idx_mapping_np) is None
                and self.sampler.can_use_small_topk_fastpath(
                    input_batch.idx_mapping_np, return_logprobs=False
                )
            ):
                use_spec = (
                    input_batch.num_draft_tokens > 0
                    and self.rejection_sampler is not None
                )
                small_k_cache = (
                    getattr(self.speculator.draft_logits, "_arc_small_k_cache", None)
                    if (
                        use_spec
                        and self.speculator is not None
                        and self.speculator.draft_logits is not None
                    )
                    else None
                )
                can_verify_cand = not use_spec or (
                    small_k_cache is not None
                    and self.rejection_sampler.watermark_key is None
                    and not self.rejection_sampler.use_block_verification
                    and self.rejection_sampler.synthetic_conditional_rates is None
                )
                if can_verify_cand:
                    cand = arc_compute_topk_candidates(
                        self.model, sample_hidden_states, ARC_SMALL_K
                    )
                    if cand is not None:
                        cand_vals, cand_idx = cand
                        if not use_spec:
                            sampler_output = self.sampler.sample_from_candidates(
                                cand_vals, cand_idx, input_batch
                            )
                        else:
                            assert small_k_cache is not None
                            sampler_output = (
                                self.rejection_sampler.verify_from_candidates(
                                    cand_vals, cand_idx, input_batch, small_k_cache
                                )
                            )
                        return (
                            sampler_output,
                            sampler_output.num_sampled,
                            sampler_output.num_rejected,
                        )
            logits = self.model.compute_logits(sample_hidden_states)""",
    ),
]

REJECTION_SAMPLER_UTILS_REPLACEMENTS = [
    (
        """    target_logit = tl.load(
        target_logits_ptr + logit_idx * target_logits_stride + token,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    target_lse = _compute_global_logsumexp(""",
        """    target_logit = tl.load(
        target_logits_ptr + logit_idx * target_logits_stride + token,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if target_logits_ptr.dtype.element_ty != tl.float32:
        target_logit = target_logit / temp
    target_lse = _compute_global_logsumexp(""",
    ),
    (
        """        # Get local target max and summed exponentials.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_max_and_sumexp(target_logits)""",
        """        # Get local target max and summed exponentials.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        if target_logits_ptr.dtype.element_ty != tl.float32:
            target_logits = target_logits / temp
        target_max, target_sumexp = _compute_max_and_sumexp(target_logits)""",
    ),
    (
        """    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)""",
        """    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if target_logits_ptr.dtype.element_ty != tl.float32:
        if temp != 0.0:
            target_logits = target_logits / temp""",
    ),
]

STAR_SHM_MARKER = "# ---- ARC_STAR_SHM_PATCH v1 ----"

ARC_STAR_SHM_MODULE_SOURCE = r'''# SPDX-License-Identifier: Apache-2.0
"""Fail-closed TP=4 Tiled Star-SHM AllReduce dispatcher for PCIe/IOMMU-isolated GPUs."""
from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
from pathlib import Path
import threading
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_RETIRED_ARENAS: list[tuple[Any, Any]] = []
_PATH_BUF_BYTES = 256


def _cpu_all_true(group: Any, is_stateless: bool, world_size: int, ok: bool) -> bool:
    """Rank-synchronized pure-CPU boolean AND collective (strictly Gloo on hybrid cpu:gloo,cuda:nccl PGs)."""
    val = 1 if ok else 0
    if is_stateless:
        flags = [
            int(group.broadcast_obj(val if int(group.rank) == r else 0, src=r))
            for r in range(int(world_size))
        ]
        return all(x == 1 for x in flags)
    flag = torch.tensor([val], dtype=torch.int32, device="cpu")
    dist.all_reduce(flag, op=ReduceOp.SUM, group=group)
    return int(flag.item()) == int(world_size)


class StarShmArenaComm:
    """Persistent `memfd` + `cudaHostRegisterMapped` Star-SHM AllReduce arena for TP=4."""

    def __init__(
        self,
        so_path: str,
        group: Any,
        device: torch.device,
        capacity: int = 192 * 5120,
        expected_sha256: str = "",
    ) -> None:
        self.device = device
        self.capacity = int(capacity)
        self.so_path = str(so_path)
        self.disabled = False
        self._root_fd: int | None = None
        self.mapping: mmap.mmap | None = None
        self.host = ctypes.c_void_p()
        self.mapped = ctypes.c_void_p()
        self.blocks = 0
        self.workers = 0
        self.pipe_d = 0
        self.bytes = 0
        self.so_sha256 = ""

        is_stateless = isinstance(group, StatelessProcessGroup)
        if is_stateless:
            self.rank = int(group.rank)
            self.world_size = int(group.world_size)
        else:
            self.rank = int(dist.get_rank(group))
            self.world_size = int(dist.get_world_size(group))
        if self.world_size != 4:
            raise ValueError(f"StarShmArenaComm requires world_size=4, got {self.world_size}")

        local_path = ""
        stage1_err = ""
        lib: Any = None
        try:
            so_bytes = Path(self.so_path).read_bytes()
            self.so_sha256 = hashlib.sha256(so_bytes).hexdigest()
            if expected_sha256 and self.so_sha256 != expected_sha256:
                raise RuntimeError(
                    f"ARC_FAST_SHM_AR sha256 mismatch: expected {expected_sha256}, got {self.so_sha256}"
                )

            lib = self.lib = ctypes.CDLL(self.so_path)
            lib.arc_bytes.argtypes = [ctypes.c_size_t]
            lib.arc_bytes.restype = ctypes.c_size_t
            lib.arc_blocks.argtypes = []
            lib.arc_blocks.restype = ctypes.c_int
            lib.arc_error.argtypes = [ctypes.c_int]
            lib.arc_error.restype = ctypes.c_char_p
            lib.arc_register.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            lib.arc_register.restype = ctypes.c_int
            lib.arc_unregister.argtypes = [ctypes.c_void_p]
            lib.arc_unregister.restype = ctypes.c_int
            lib.arc_launch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            lib.arc_launch.restype = ctypes.c_int
            if hasattr(lib, "arc_init_workers"):
                lib.arc_init_workers.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                lib.arc_init_workers.restype = ctypes.c_int
            if hasattr(lib, "arc_workers"):
                lib.arc_workers.argtypes = []
                lib.arc_workers.restype = ctypes.c_int
                self.workers = int(lib.arc_workers())
            if hasattr(lib, "arc_pipe_d"):
                lib.arc_pipe_d.argtypes = []
                lib.arc_pipe_d.restype = ctypes.c_int
                self.pipe_d = int(lib.arc_pipe_d())

            self.blocks = int(lib.arc_blocks())
            self.bytes = int(lib.arc_bytes(self.capacity))
            if self.rank == 0:
                self._root_fd = os.memfd_create("arc_vllm_star_shm", 0)
                os.ftruncate(self._root_fd, self.bytes)
                if hasattr(os, "posix_fallocate"):
                    os.posix_fallocate(self._root_fd, 0, self.bytes)
                raw_check = f"/proc/{os.getpid()}/fd/{self._root_fd}".encode("utf-8")
                if len(raw_check) >= _PATH_BUF_BYTES:
                    raise RuntimeError("memfd procfs path exceeded fixed broadcast buffer")
                local_path = raw_check.decode("utf-8")
        except Exception as exc:
            stage1_err = str(exc)
            local_path = ""

        # Collective #1: broadcast share_path (empty string if rank 0 failed stage 1).
        if is_stateless:
            share_path = str(group.broadcast_obj(local_path if self.rank == 0 else "", src=0) or "")
        else:
            raw = local_path.encode("utf-8") if self.rank == 0 else b""
            buf = torch.zeros(_PATH_BUF_BYTES, dtype=torch.uint8, device="cpu")
            if self.rank == 0 and raw:
                buf[: len(raw)] = torch.tensor(list(raw), dtype=torch.uint8, device="cpu")
            ranks = dist.get_process_group_ranks(group)
            dist.broadcast(buf, src=ranks[0], group=group)
            raw_list = [int(b) for b in buf.tolist() if int(b) != 0]
            share_path = bytes(raw_list).decode("utf-8", errors="replace")

        stage2_ok = False
        stage2_err = stage1_err
        if not stage1_err and share_path and lib is not None:
            try:
                open_fd = os.open(share_path, os.O_RDWR)
                try:
                    self.mapping = mmap.mmap(
                        open_fd,
                        self.bytes,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(open_fd)

                assert self.mapping is not None
                self.anchor = ctypes.c_char.from_buffer(self.mapping)
                self.host = ctypes.c_void_p(ctypes.addressof(self.anchor))
                self.mapped = ctypes.c_void_p()
                with torch.cuda.device(self.device):
                    self._check(
                        lib.arc_register(self.host, self.bytes, ctypes.byref(self.mapped)),
                        "arc_register",
                    )
                    if hasattr(lib, "arc_init_workers"):
                        self._check(
                            lib.arc_init_workers(
                                self.host,
                                ctypes.c_size_t(self.bytes),
                                ctypes.c_size_t(self.capacity),
                                int(self.rank),
                            ),
                            "arc_init_workers",
                        )
                    self.seq = torch.zeros(self.blocks, device=self.device, dtype=torch.int64)
                stage2_ok = True
            except Exception as exc:
                stage2_err = str(exc)
                stage2_ok = False

        # Collective #2: pure-CPU Gloo barrier + all-ranks stage2 health check before closing _root_fd or launching GPU kernel.
        all_stage2_ok = _cpu_all_true(group, is_stateless, self.world_size, stage2_ok)
        if self._root_fd is not None:
            try:
                os.close(self._root_fd)
            except Exception:
                pass
            self._root_fd = None

        if not all_stage2_ok:
            self.destroy()
            raise RuntimeError(stage2_err or "peer_rank_stage2_init_failed")

        stage3_ok = False
        stage3_err = ""
        try:
            with torch.cuda.device(self.device):
                warm = torch.zeros(8, device=self.device, dtype=torch.bfloat16)
                self.all_reduce(warm, warm, stream=current_stream())
                torch.cuda.synchronize(self.device)
                del warm
            stage3_ok = True
        except Exception as exc:
            stage3_err = str(exc)
            stage3_ok = False

        # Collective #3: pure-CPU Gloo all-ranks warmup verification.
        all_stage3_ok = _cpu_all_true(group, is_stateless, self.world_size, stage3_ok)
        if not all_stage3_ok:
            self.destroy()
            raise RuntimeError(stage3_err or "peer_rank_warmup_failed")

    def _check(self, code: int, fn_name: str) -> None:
        if code != 0:
            err_ptr = self.lib.arc_error(code)
            msg = err_ptr.decode("utf-8", errors="replace") if err_ptr else f"code={code}"
            raise RuntimeError(f"ARC_FAST_SHM_AR {fn_name} failed ({code}): {msg}")

    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor | None = None,
        stream: Any = None,
    ) -> torch.Tensor:
        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)
        if stream is None:
            stream = current_stream()
        self._check(
            self.lib.arc_launch(
                self.mapped,
                ctypes.c_void_p(self.seq.data_ptr()),
                ctypes.c_void_p(in_tensor.data_ptr()),
                ctypes.c_void_p(out_tensor.data_ptr()),
                ctypes.c_size_t(in_tensor.numel()),
                ctypes.c_size_t(self.capacity),
                int(self.rank),
                ctypes.c_void_p(stream.cuda_stream),
            ),
            "arc_launch",
        )
        return out_tensor

    def destroy(self) -> None:
        if self.disabled:
            return
        self.disabled = True
        host_ptr = getattr(self, "host", None)
        if host_ptr is not None and host_ptr.value:
            self.host = ctypes.c_void_p()
            unreg_ok = [False]

            def _unreg() -> None:
                try:
                    with torch.cuda.device(self.device):
                        torch.cuda.synchronize(self.device)
                        rc = int(self.lib.arc_unregister(host_ptr))
                        unreg_ok[0] = (rc == 0)
                except Exception:
                    unreg_ok[0] = False

            t = threading.Thread(target=_unreg, daemon=True)
            t.start()
            t.join(timeout=2.0)
            if unreg_ok[0]:
                if hasattr(self, "anchor"):
                    del self.anchor
                if self.mapping is not None:
                    try:
                        self.mapping.close()
                    except Exception:
                        pass
                    self.mapping = None
            else:
                # Keep virtual pages alive until OS process teardown if CUDAGraphs still reference them.
                _RETIRED_ARENAS.append((getattr(self, "mapping", None), getattr(self, "anchor", None)))
        else:
            if hasattr(self, "anchor"):
                del self.anchor
            if self.mapping is not None:
                try:
                    self.mapping.close()
                except Exception:
                    pass
                self.mapping = None
        if self._root_fd is not None:
            try:
                os.close(self._root_fd)
            except Exception:
                pass
            self._root_fd = None


def _has_partial_overlap(x: torch.Tensor, y: torch.Tensor) -> bool:
    x_ptr = x.data_ptr()
    y_ptr = y.data_ptr()
    if x_ptr == y_ptr:
        return False
    nbytes = x.numel() * x.element_size()
    return (x_ptr < y_ptr + nbytes) and (y_ptr < x_ptr + nbytes)


def maybe_attach_arc_star_shm(owner: Any) -> bool:
    so_path = os.environ.get("ARC_FAST_SHM_AR_SO", "").strip()
    if not so_path:
        return False
    if owner.unique_name.split(":", 1)[0] != "tp" or owner.world_size != 4:
        return False
    comm = getattr(owner, "pynccl_comm", None)
    if comm is None or comm.disabled:
        logger.warning(
            "ARC_FAST_SHM_AR fallback_pynccl group=%s rank=%d/%d reason=pynccl_unavailable",
            owner.unique_name,
            int(getattr(owner, "rank", -1)),
            int(owner.world_size),
        )
        return False

    try:
        expected_sha = os.environ.get("ARC_FAST_SHM_AR_SHA256", "").strip()
        capacity = int(os.environ.get("ARC_FAST_SHM_AR_CAPACITY", str(192 * 5120)))
        min_numel = int(os.environ.get("ARC_FAST_SHM_AR_MIN_NUMEL", "1"))
        max_numel = int(os.environ.get("ARC_FAST_SHM_AR_MAX_NUMEL", str(6144 * 5120)))
        if min_numel <= 0 or max_numel < min_numel:
            raise ValueError(f"Invalid ARC_FAST_SHM_AR numel range [{min_numel}, {max_numel}]")
    except Exception as exc:
        logger.warning(
            "ARC_FAST_SHM_AR fallback_pynccl group=%s rank=%d/%d reason=%s",
            owner.unique_name,
            int(getattr(comm, "rank", getattr(owner, "rank", -1))),
            int(owner.world_size),
            exc,
        )
        return False

    coord_group = comm.group if getattr(comm, "group", None) is not None else owner.cpu_group
    try:
        star = StarShmArenaComm(
            so_path=so_path,
            group=coord_group,
            device=comm.device,
            capacity=capacity,
            expected_sha256=expected_sha,
        )
    except Exception as exc:
        logger.warning(
            "ARC_FAST_SHM_AR fallback_pynccl group=%s rank=%d/%d reason=%s",
            owner.unique_name,
            int(getattr(comm, "rank", getattr(owner, "rank", -1))),
            int(owner.world_size),
            exc,
        )
        return False

    # Ensure no optional TP AllReduce backend intercepts ahead of pynccl_comm.
    for attr in ("ca_comm", "symm_mem_comm", "fi_ar_comm", "fi_pcie_ipc_ar_comm", "qr_comm", "aiter_ar_comm"):
        sub = getattr(owner, attr, None)
        if sub is not None and hasattr(sub, "disabled"):
            sub.disabled = True

    native = comm.all_reduce

    def _arc_dispatched_all_reduce(
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor | None = None,
        op: ReduceOp = ReduceOp.SUM,
        stream: Any = None,
    ) -> torch.Tensor | None:
        if comm.disabled or star.disabled:
            return native(in_tensor, out_tensor, op, stream)
        if in_tensor.device != comm.device:
            raise AssertionError(
                f"communicator device {comm.device} != input tensor device {in_tensor.device}"
            )
        y = out_tensor
        if y is not None and (
            y.device != in_tensor.device
            or y.dtype != in_tensor.dtype
            or y.shape != in_tensor.shape
        ):
            raise ValueError("Invalid out_tensor metadata for AllReduce")

        n_elem = in_tensor.numel()
        # Rank-consistent collective routing: depends ONLY on group-invariant (op, dtype, numel).
        eligible = (
            op == ReduceOp.SUM
            and in_tensor.dtype == torch.bfloat16
            and min_numel <= n_elem <= max_numel
        )

        cur_s = current_stream()
        s = cur_s if stream is None else stream
        strided_or_overlap = (
            not in_tensor.is_contiguous()
            or (y is not None and (not y.is_contiguous() or _has_partial_overlap(in_tensor, y)))
        )

        if not strided_or_overlap and s == cur_s:
            if eligible:
                return star.all_reduce(in_tensor, y, stream=s)
            return native(in_tensor, y, op, s)

        with torch.cuda.stream(s):
            if strided_or_overlap:
                packed = in_tensor.contiguous()
                if eligible:
                    tmp = star.all_reduce(packed, None, stream=s)
                else:
                    tmp = torch.empty_like(packed)
                    native(packed, tmp, op, s)
                if y is None:
                    return tmp
                y.copy_(tmp)
                return y
            if eligible:
                return star.all_reduce(in_tensor, y, stream=s)
            return native(in_tensor, y, op, s)

    comm.all_reduce = _arc_dispatched_all_reduce
    owner._arc_star_shm = star
    logger.info(
        "ARC_FAST_SHM_AR attached group=%s rank=%d/%d blocks=%d workers=%d pipe_d=%d capacity=%d range=[%d,%d] shm_mib=%.2f so=%s",
        owner.unique_name,
        star.rank,
        star.world_size,
        star.blocks,
        star.workers,
        star.pipe_d,
        star.capacity,
        min_numel,
        max_numel,
        star.bytes / (1024.0**2),
        star.so_path,
    )
    return True


def maybe_destroy_arc_star_shm(owner: Any) -> None:
    star = getattr(owner, "_arc_star_shm", None)
    if star is not None:
        owner._arc_star_shm = None
        star.destroy()
'''

CUDA_COMMUNICATOR_REPLACEMENTS = [
    (
        """from .aiter_custom_all_reduce import AiterCustomAllreduce
from .base_device_communicator import DeviceCommunicatorBase""",
        """from .aiter_custom_all_reduce import AiterCustomAllreduce
from .arc_star_shm import maybe_attach_arc_star_shm, maybe_destroy_arc_star_shm
from .base_device_communicator import DeviceCommunicatorBase""",
    ),
    (
        """        if self.world_size > 1:
            self._log_all_reduce_backend_selection()""",
        """        maybe_attach_arc_star_shm(self)

        if self.world_size > 1:
            self._log_all_reduce_backend_selection()""",
    ),
    (
        """    def destroy(self):
        if self.pynccl_comm is not None:
            self.pynccl_comm.destroy()
            self.pynccl_comm = None""",
        """    def destroy(self):
        if self.pynccl_comm is not None:
            self.pynccl_comm.destroy()
            self.pynccl_comm = None
        maybe_destroy_arc_star_shm(self)""",
    ),
]


def apply_replacements(source, replacements, marker, label):
    if marker in source:
        return source
    out = source
    for idx, (old, new) in enumerate(replacements):
        if old not in out:
            raise RuntimeError(f"{label} replacement anchor #{idx} missing")
        out = out.replace(old, new, 1)
    return out.rstrip("\n") + "\n\n" + marker + "\n"


def locate(venv):
    hits = sorted(Path(venv).glob("lib/python3*/site-packages/vllm/v1/core/single_type_kv_cache_manager.py"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one vLLM single_type_kv_cache_manager.py under {venv}, found {hits}")
    return hits[0]


def vllm_version(module_path):
    site = module_path.parents[3]
    metas = sorted(site.glob("vllm-*.dist-info/METADATA"))
    for meta in metas:
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    return None


def patch_source(source):
    if MARKER in source:
        raise RuntimeError("ARC Mamba pin patch already installed")
    missing = [a.splitlines()[0].strip() for a in ANCHORS if a not in source]
    if missing:
        raise RuntimeError(f"Upstream anchors missing; refusing to patch: {missing}")
    return source.rstrip("\n") + "\n\n\n" + PATCH


def patch_file_with_replacements(path, replacements, marker, label):
    orig_path = path.with_suffix(path.suffix + ".orig")
    current = path.read_text(encoding="utf-8")
    if orig_path.exists():
        source = orig_path.read_text(encoding="utf-8")
    else:
        source = current
        if marker not in source:
            orig_path.write_text(source, encoding="utf-8")
    patched = apply_replacements(source, replacements, marker, label)
    compile(patched, str(path), "exec")
    path.unlink(missing_ok=True)
    path.write_text(patched, encoding="utf-8")
    for pyc in path.parent.glob(f"__pycache__/{path.stem}*.pyc"):
        pyc.unlink()
    return {
        "file": str(path),
        "sha256_before": sha256(current),
        "sha256_after": sha256(patched),
        "already_patched": marker in current,
    }


def install(venv, receipt_path):
    path = locate(venv)
    version = vllm_version(path)
    if version is None or not version.startswith(EXPECTED_VLLM):
        raise RuntimeError(f"Expected vLLM {EXPECTED_VLLM}, found {version}")
    orig_path = path.with_suffix(path.suffix + ".orig")
    current = path.read_text(encoding="utf-8")
    if orig_path.exists():
        source = orig_path.read_text(encoding="utf-8")
    else:
        source = current
        if MARKER not in source:
            orig_path.write_text(source, encoding="utf-8")
    patched = source if MARKER in source else patch_source(source)
    compile(patched, str(path), "exec")
    path.unlink(missing_ok=True)
    path.write_text(patched, encoding="utf-8")
    for pyc in path.parent.glob("__pycache__/single_type_kv_cache_manager*.pyc"):
        pyc.unlink()

    v1_dir = path.parents[1]
    vllm_dir = path.parents[2]
    small_topk_module_path = v1_dir / "worker/gpu/sample/arc_small_topk.py"
    compile(ARC_SMALL_TOPK_MODULE_SOURCE, str(small_topk_module_path), "exec")
    small_topk_module_path.unlink(missing_ok=True)
    small_topk_module_path.write_text(ARC_SMALL_TOPK_MODULE_SOURCE, encoding="utf-8")
    for pyc in small_topk_module_path.parent.glob("__pycache__/arc_small_topk*.pyc"):
        pyc.unlink()

    arc_star_shm_module_path = vllm_dir / "distributed/device_communicators/arc_star_shm.py"
    compile(ARC_STAR_SHM_MODULE_SOURCE, str(arc_star_shm_module_path), "exec")
    arc_star_shm_module_path.unlink(missing_ok=True)
    arc_star_shm_module_path.write_text(ARC_STAR_SHM_MODULE_SOURCE, encoding="utf-8")
    for pyc in arc_star_shm_module_path.parent.glob("__pycache__/arc_star_shm*.pyc"):
        pyc.unlink()

    cuda_communicator_path = vllm_dir / "distributed/device_communicators/cuda_communicator.py"
    triton_helpers_path = v1_dir / "attention/ops/triton_attention_helpers.py"
    triton_unified_path = v1_dir / "attention/ops/triton_unified_attention.py"
    triton_attn_path = v1_dir / "attention/backends/triton_attn.py"
    model_runner_path = v1_dir / "worker/gpu/model_runner.py"
    sampler_path = v1_dir / "worker/gpu/sample/sampler.py"
    speculator_path = v1_dir / "worker/gpu/spec_decode/speculator.py"
    rejection_sampler_path = v1_dir / "worker/gpu/spec_decode/rejection_sampler.py"
    rejection_sampler_utils_path = v1_dir / "worker/gpu/spec_decode/rejection_sampler_utils.py"
    triton_receipts = {
        "arc_small_topk_module": {
            "file": str(small_topk_module_path),
            "sha256_after": sha256(ARC_SMALL_TOPK_MODULE_SOURCE),
        },
        "arc_star_shm_module": {
            "file": str(arc_star_shm_module_path),
            "sha256_after": sha256(ARC_STAR_SHM_MODULE_SOURCE),
        },
        "cuda_communicator": patch_file_with_replacements(
            cuda_communicator_path,
            CUDA_COMMUNICATOR_REPLACEMENTS,
            STAR_SHM_MARKER,
            "cuda_communicator",
        ),
        "triton_attention_helpers": patch_file_with_replacements(
            triton_helpers_path,
            TRITON_HELPERS_REPLACEMENTS,
            TRITON_HELPERS_MARKER,
            "triton_attention_helpers",
        ),
        "triton_unified_attention": patch_file_with_replacements(
            triton_unified_path,
            TRITON_UNIFIED_REPLACEMENTS,
            TRITON_HELPERS_MARKER,
            "triton_unified_attention",
        ),
        "triton_attn": patch_file_with_replacements(
            triton_attn_path,
            TRITON_ATTN_REPLACEMENTS,
            TRITON_HELPERS_MARKER,
            "triton_attn",
        ),
        "model_runner": patch_file_with_replacements(
            model_runner_path,
            MODEL_RUNNER_REPLACEMENTS,
            SMALL_TOPK_MARKER,
            "model_runner",
        ),
        "sampler": patch_file_with_replacements(
            sampler_path,
            SAMPLER_REPLACEMENTS,
            SMALL_TOPK_MARKER,
            "sampler",
        ),
        "speculator": patch_file_with_replacements(
            speculator_path,
            SPECULATOR_REPLACEMENTS,
            SMALL_TOPK_MARKER,
            "speculator",
        ),
        "rejection_sampler": patch_file_with_replacements(
            rejection_sampler_path,
            REJECTION_SAMPLER_REPLACEMENTS,
            SMALL_TOPK_MARKER,
            "rejection_sampler",
        ),
        "rejection_sampler_utils": patch_file_with_replacements(
            rejection_sampler_utils_path,
            REJECTION_SAMPLER_UTILS_REPLACEMENTS,
            TRITON_HELPERS_MARKER,
            "rejection_sampler_utils",
        ),
    }

    receipt = {
        "patch": "ARC_MAMBA_PIN_PATCH v1 + ARC_TRITON_ATTN_PATCH v2 + ARC_SMALL_TOPK_PATCH v4 + ARC_STAR_SHM_PATCH v1",
        "file": str(path),
        "vllm": version,
        "sha256_before": sha256(source),
        "sha256_after": sha256(patched),
        "already_patched": MARKER in source,
        "runtime_disable": "ARC_MAMBA_PIN=0 / ARC_SMALL_TOPK=0 / ARC_TP_LOCAL_TOPK=0 / unset ARC_FAST_SHM_AR_SO",
        "max_per_request_env": "ARC_MAMBA_PIN_MAX_PER_REQ (default 1)",
        "triton_attn_patches": triton_receipts,
    }
    Path(receipt_path).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    install(sys.argv[1], sys.argv[2])
