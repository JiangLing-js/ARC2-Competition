// /tokenize is already needed for the remaining context limit. Reuse its token
// IDs to measure stable history; no hash, extra tokenization, or model request.
export class PrefixAccounting {
  constructor() {
    this.previous = null;
    this.pending = null;
    this.totals = { cold_input_tokens: 0, continuation_input_tokens: 0,
      continuation_cached_tokens: 0, reusable_previous_input_tokens: 0,
      reused_previous_input_tokens: 0, previous_input_prefix_mismatch_tokens: 0,
      prefix_comparison_unavailable: 0 };
  }

  prepare({ count, tokens }) {
    const continuation = this.pending !== null;
    let common = null;
    if (this.previous && Array.isArray(tokens)) {
      common = 0;
      while (common < Math.min(this.previous.length, tokens.length) && this.previous[common] === tokens[common]) common++;
    }
    this.pending = { continuation, prompt_tokens: count,
      previous_prompt_tokens: this.previous?.length ?? null,
      reusable_previous_input_tokens: common,
      previous_input_prefix_mismatch_tokens: common === null ? null : this.previous.length - common };
    this.previous = Array.isArray(tokens) ? tokens : null;
    return this.pending;
  }

  settle(usage) {
    const prompt = usage.input + usage.cacheRead + usage.cacheWrite;
    const receipt = { ...this.pending, input_tokens: prompt, cached_tokens: usage.cacheRead,
      output_tokens: usage.output, reasoning_tokens: usage.reasoning ?? 0 };
    if (this.pending.continuation) {
      this.totals.continuation_input_tokens += prompt;
      this.totals.continuation_cached_tokens += usage.cacheRead;
      const common = this.pending.reusable_previous_input_tokens;
      if (common === null) this.totals.prefix_comparison_unavailable++;
      else {
        this.totals.reusable_previous_input_tokens += common;
        this.totals.reused_previous_input_tokens += Math.min(common, usage.cacheRead);
        this.totals.previous_input_prefix_mismatch_tokens += this.pending.previous_input_prefix_mismatch_tokens;
      }
    } else this.totals.cold_input_tokens += prompt;
    return receipt;
  }
}
