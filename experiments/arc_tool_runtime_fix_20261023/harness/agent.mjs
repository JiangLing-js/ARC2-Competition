import { runAgentLoop } from '@earendil-works/pi-agent-core';
import { stream as streamQwen } from '@earendil-works/pi-ai/api/openai-completions';
import { Type } from 'typebox';
import { reservationTokens } from './kv-scheduler.mjs';
import { PrefixAccounting } from './prefix-accounting.mjs';
import { isBackendFailure } from './failure-state.mjs';
import { selectAttempts } from './candidate-pool.mjs';

export const SUBMIT_DESCRIPTION = 'Check a complete transform(grid) program in a fresh workspace on every demo and test input.';
export const TRANSFORM_ALT_NOTE = ' If a different rule also explains every example, you may additionally define transform_alt(grid) in the same program; it is checked separately and never changes the result of transform. Do not search for an alternative only for this.';

export const SYSTEM = `Solve the ARC grid transformation using the Python tools.
Infer one general rule from the training pairs, investigate it with Python, then submit a complete program.
Start with a compact Python inspection of every training pair: input/output shapes, changed colors or cells, and the object properties relevant to the transformation.
Use the python tool for concrete questions: inspect shapes, count colors, locate objects, or test a hypothesis.
When hypotheses compete, run the smallest experiment that distinguishes them. Compute transformations and comparisons in Python rather than simulating arrays in prose.
Implement the simplest rule consistent with all examples. Infer its parameters from the input; avoid memorizing particular example grids.
Print the small observations you need. Use verify_demos(transform) to test a proposed rule before submission.
If an example fails, inspect its first concrete mismatch and revise the responsible part of the rule while retaining what already works.
The Python workspace persists between python calls. It contains:
- np, math, Counter, defaultdict, deque
- train_pairs: a list of dictionaries. Each pair has pair['input'] and pair['output'], both list-of-lists grids.
- test_inputs: a list of input grids, with no answers.
- verify_demos(transform): returns comparisons on the training examples.
For example: print([(np.shape(p['input']), np.shape(p['output'])) for p in train_pairs]).
Submit with submit_program(code=...) when ready. The code must define transform(grid), returning a rectangular integer grid of colors 0..9.
The final checker runs that complete program in a fresh workspace; include every function and constant it needs.
np, math, Counter, defaultdict and deque are also available there. Imports from numpy, math, collections, itertools, functools, copy and heapq are supported.
Use array operations or simple Python. File, network, process access and reflection are unavailable.
There are at most 12 Python experiments and 3 final submissions. A failed submission returns concrete demo feedback so you can repair the program.
Stop at the first program that passes every training example and produces legal test grids.
Once verify_demos passes all examples, submit the self-contained program instead of exploring more alternatives.
Call tools through their native function interface. Put Python source in the code argument, without markdown fences.`;

const STREAM_TIMEOUT_MS = 1800_000;
export const FINALIZE_PROMPT = 'Call submit_program(code=...) now with your best complete, self-contained transform(grid) implementation.';
// pi-ai 1.3 drops reasoning-only assistant messages. A private marker keeps the
// message through its serializer; onPayload removes it before tokenization/wire.
const REASONING_ONLY_MARKER = '__ARC_PRESERVE_REASONING_ONLY_7C681A__';
export function preserveReasoningOnlyMessages(messages) {
  return messages.map(message => {
    if (message.role !== 'assistant' || !Array.isArray(message.content)) return message;
    const thinking = message.content.filter(c => c.type === 'thinking' && c.thinking?.trim());
    if (!thinking.length || message.content.some(c => c.type === 'toolCall' || (c.type === 'text' && c.text?.length))) return message;
    return { ...message, content: [
      ...message.content.map(c => c.type === 'thinking' ? { ...c, thinkingSignature: 'reasoning_content' } : c),
      { type: 'text', text: REASONING_ONLY_MARKER },
    ] };
  });
}
export function restoreReasoningOnlyPayload(messages) {
  return messages.map(message => message.role === 'assistant' && message.content === REASONING_ONLY_MARKER
    ? { ...message, content: '', reasoning_content: message.reasoning_content ?? message.reasoning ?? '' } : message);
}
export function isTransientSocketError(error) {
  const detail = `${String(error ?? '')} ${String(error?.cause ?? '')} ${String(error?.cause?.code ?? '')}`;
  return /fetch failed|ECONNRESET|EPIPE|UND_ERR_SOCKET|socket hang up|other side closed/i.test(detail)
    && !/ECONNREFUSED/i.test(detail);
}

let longDispatcher = null;
async function ensureLongTimeoutDispatcher() {
  if (longDispatcher) return longDispatcher;
  const sym = Symbol.for('undici.globalDispatcher.1');
  if (!globalThis[sym]) {
    await fetch('http://127.0.0.1:1').catch(() => {});
  }
  const Agent = globalThis[sym]?.constructor;
  if (typeof Agent === 'function') {
    longDispatcher = new Agent({
      headersTimeout: STREAM_TIMEOUT_MS,
      bodyTimeout: STREAM_TIMEOUT_MS,
      connectTimeout: 60_000,
      keepAliveTimeout: 1_000,
      keepAliveMaxTimeout: 2_000,
    });
    globalThis[sym] = longDispatcher;
  }
  return longDispatcher;
}

export async function resilientFetch(url, init, { fetchImpl = fetch, retryDelaysMs = [120, 400, 1000] } = {}) {
  const dispatcher = fetchImpl === fetch ? await ensureLongTimeoutDispatcher() : null;
  const reqInit = dispatcher ? { ...init, dispatcher } : init;
  for (let attempt = 0; ; attempt++) {
    try {
      return await fetchImpl(url, reqInit);
    } catch (error) {
      if (attempt >= retryDelaysMs.length || init?.signal?.aborted || !isTransientSocketError(error)) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, retryDelaysMs[attempt]));
      if (init?.signal?.aborted) throw error;
    }
  }
}

const longTimeoutFetch = (url, init) => resilientFetch(url, init);

export function localModel(baseUrl, id = 'gr48k') {
  return { id, name: 'Qwen3.8-27B GR48K FP8_DYNAMIC', provider: 'arc-local',
    api: 'openai-completions', baseUrl, reasoning: true, input: ['text'],
    contextWindow: 131072, maxTokens: 65000,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { supportsStore: false, supportsDeveloperRole: false,
      supportsReasoningEffort: false, maxTokensField: 'max_tokens',
      thinkingFormat: 'qwen-chat-template' } };
}

export function buildInitialUserContent(task) {
  return JSON.stringify(task);
}

// Pi owns the conversation and tool-result protocol; this adapter only supplies
// Qwen's template controls and the shared task budget.
export async function solveTask({ task, model, bridge, signal, emit = async () => {},
  outputLimit = 65000, pythonLimit = 24, pythonNoticeAfter = 16, tokenize = true, scheduler, governor = null,
  taskId, reasoningEffort = 'medium', transformAlt = false, rewriteSystemPythonLimit = false,
  seed = 42, wave = 1, governorDecision = null,
  turn1ThinkCap = 0, turnThinkCap = 0, codeReserve = 4096, finalizeReserve = 0,
  previousPredictions = null, duplicateExtendTokens = 0, onBackendFailure = () => {},
  emergencyFinalizeTokens = 4096, emergencyRepairTokens = 2048, lateBudgetNoticeTokens = 0 }) {
  if (!Number.isInteger(pythonLimit) || pythonLimit < 1 || pythonLimit > 30) {
    throw new Error('pythonLimit must be an integer from 1 to 30');
  }
  if (!['xhigh', 'medium', 'low'].includes(reasoningEffort)) {
    throw new Error(`Qwen3.8 supports xhigh, medium and low; received ${reasoningEffort}`);
  }
  for (const [name, value] of Object.entries({ emergencyFinalizeTokens, emergencyRepairTokens, lateBudgetNoticeTokens })) {
    if (!Number.isInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer`);
  }
  const initialDecision = governorDecision ?? (governor?.active?.has(taskId)
    ? { limit: governor.active.get(taskId).limit, reduced: governor.active.get(taskId).limit < outputLimit } : null);
  if (initialDecision) outputLimit = Math.min(outputLimit, initialDecision.limit);
  const started = Date.now();
  const prefix = new PrefixAccounting(), receipts = [], streamDiagnostics = [];
  const state = { status: 'running', eligible: false, seed, wave, reasoning_effort: reasoningEffort, output_tokens: 0, input_tokens: 0,
    cached_input_tokens: 0, reasoning_tokens: 0, requests: 0, python_calls: 0,
    python_limit: pythonLimit, output_limit: outputLimit, governor_decision: initialDecision, final_checks: 0, tool_seconds: 0, tool_errors: 0, workspace_resets: 0, unknown_usage_requests: 0,
    predictions: [], fallback_predictions: [], fallback_demo_pass_count: 0, fallback_demo_diff_cells: null,
    verified_fallback_predictions: [], verified_fallback_used: false, answer_source: null,
    best_demo_pass_count: 0, routed_final_checks: 0, overflow_probes_attempted: 0, seed_jitter_active: false,
    finalize_state: 'EXPLORE', emergency_tokens_granted: 0, emergency_output_tokens: 0,
    exploration_output_limit: outputLimit, emergency_repair_used: false,
    stop_reason: null, admission_wait_seconds: 0, first_dispatch_at: null,
    phase: 'preparing', cache: prefix.totals };
  let pendingCorrection = false, corrections = 0, accountedRequests = 0;
  let requestPromptTokens = 0, requestOutputAllowance = 0;
  let finalizeState = 'EXPLORE', requestReservedFinalize = false, requestLimitReason = null;
  let currentFinalizeAllowance = 0, requestWasFinalize = false, emergencySubmitFailedDemo = false;
  let emergencySubmissionsThisRequest = 0, lateNoticeSent = false;
  let pendingFinalizePrompt = false, savedEligibleResult = null, duplicateExtendedOnce = false;
  let firstEligibleSubmissionSeen = false, lastTurnHadTools = null;
  const eligibleCandidates = [];
  const setFinalizeState = value => { finalizeState = value; state.finalize_state = value; };
  const canFinalize = () => !state.eligible && state.final_checks < 3 && finalizeState === 'EXPLORE';
  const grantTokens = (tokens, reason) => {
    if (tokens <= 0) return 0;
    const decision = governor?.grantExtension?.(taskId, tokens, { wave, reason });
    // A governor without the extension API cannot authorize extra obligations.
    const granted = Math.max(0, Math.min(tokens, governor
      ? (typeof decision === 'number' ? decision : decision?.granted_tokens ?? 0) : tokens));
    outputLimit += granted; state.output_limit = outputLimit;
    state.emergency_tokens_granted += granted;
    (state.emergency_grants ??= []).push({ reason, requested_tokens: tokens, granted_tokens: granted, decision: decision ?? null });
    return granted;
  };
  const requestFinalize = (reason, addPrompt = false, reserved = false) => {
    if (!canFinalize()) return false;
    if (!reserved && emergencyFinalizeTokens <= 0) return false;
    currentFinalizeAllowance = reserved ? finalizeReserve : emergencyFinalizeTokens;
    if (!reserved) grantTokens(emergencyFinalizeTokens, reason);
    if (outputLimit - state.output_tokens <= 0) return false;
    setFinalizeState('PENDING'); state.finalize_used = true; state.finalize_reason = reason;
    state.finalize_reserved = reserved;
    pendingFinalizePrompt = addPrompt; pendingCorrection = false;
    return true;
  };
  const requestRepair = () => {
    if (!emergencySubmitFailedDemo || state.eligible || state.final_checks >= 3 || state.emergency_repair_used
      || emergencyRepairTokens <= 0) return false;
    // Reuse unspent allowance; replenish only when it cannot support a repair.
    if (outputLimit - state.output_tokens < 1536) grantTokens(emergencyRepairTokens, 'finalize_repair');
    if (outputLimit - state.output_tokens < 1536) return false;
    state.emergency_repair_used = true; currentFinalizeAllowance = emergencyRepairTokens;
    pendingCorrection = false; setFinalizeState('REPAIR_PENDING');
    return true;
  };
  const liveUsage = tokens => {
    if (governor?.updateLiveUsage) governor.updateLiveUsage(taskId, tokens);
    else governor?.progress(taskId, tokens);
  };
  let activeStreamTiming = null;
  const settled = new WeakSet();
  const settle = (message) => {
    if (message.role !== 'assistant' || settled.has(message) || state.requests === accountedRequests) return;
    settled.add(message);
    accountedRequests = state.requests;
    const u = message.usage;
    state.output_tokens += u.output;
    if (requestWasFinalize) state.emergency_output_tokens += u.output;
    state.input_tokens += u.input + u.cacheRead + u.cacheWrite;
    state.cached_input_tokens += u.cacheRead;
    state.reasoning_tokens += u.reasoning ?? 0; // Already included in output_tokens.
    if (state.requests === 1) state.turn1_output_tokens = u.output;
    liveUsage(state.output_tokens);
    if (!u.totalTokens) state.unknown_usage_requests++;
    if (prefix.pending && u.totalTokens) receipts.push({ type: 'cache_receipt',
      request_number: state.requests, at: Date.now(), ...prefix.settle(u) });
    if (activeStreamTiming && (message.stopReason === 'error' || message.stopReason === 'aborted' || !u.totalTokens)) {
      const nowMs = Date.now();
      streamDiagnostics.push({
        type: 'stream_diagnostic',
        request_number: state.requests,
        at: nowMs,
        stop_reason: message.stopReason,
        error_message: message.errorMessage ?? null,
        duration_ms: nowMs - activeStreamTiming.dispatchedAt,
        headers_ms: activeStreamTiming.headersAt ? activeStreamTiming.headersAt - activeStreamTiming.dispatchedAt : null,
        first_chunk_ms: activeStreamTiming.firstChunkAt ? activeStreamTiming.firstChunkAt - activeStreamTiming.dispatchedAt : null,
        last_chunk_gap_ms: activeStreamTiming.lastChunkAt ? nowMs - activeStreamTiming.lastChunkAt : null,
        chunk_count: activeStreamTiming.chunkCount,
      });
    }
  };
  const rememberVerified = result => {
    if (result?.verified_fallback_predictions?.length) {
      state.verified_fallback_predictions = structuredClone(result.verified_fallback_predictions);
      state.verified_fallback_action_id = result.verified_fallback_action_id ?? result.action_id;
    }
  };
  const tools = ['python', 'submit_program'].map(name => ({
    name, label: name, executionMode: 'sequential',
    description: name === 'python' ? 'Execute Python in the persistent task workspace. Print observations.'
      : SUBMIT_DESCRIPTION + (transformAlt ? TRANSFORM_ALT_NOTE : ''),
    parameters: Type.Object({ code: Type.String() }, { additionalProperties: false }),
    async execute(id, { code }) {
      const routed = name === 'python' && state.python_calls >= pythonLimit;
      if (routed) state.overflow_probes_attempted++;
      else name === 'python' ? state.python_calls++ : state.final_checks++;
      if (requestWasFinalize && name === 'submit_program') emergencySubmissionsThisRequest++;
      let result;
      try { result = await bridge.call({ id, name: routed ? 'python_candidate' : name, code }); }
      catch (error) {
        rememberVerified(bridge.verifiedSnapshot);
        state.status = signal?.aborted ? 'deadline' : 'task_error';
        state.stop_reason = String(error); state.tool_errors++;
        return { content: [{ type: 'text', text: state.stop_reason }],
          details: { transport_closed: true }, isError: true, terminate: true };
      }
      if (routed && result.eligible && result.final_check_consumed) {
        state.final_checks++; state.routed_final_checks++;
      }
      rememberVerified(result);
      if (routed && !result.eligible) requestFinalize('overflow_probe_failed');
      if (result.stagnation_detected) {
        state.seed_jitter_active = true; state.stagnation_count = result.stagnation_count ?? 1;
      }
      state.best_demo_pass_count = Math.max(state.best_demo_pass_count, result.best_demo_pass_count ?? 0, result.demo_pass_count ?? 0);
      if (!result.eligible && ((result.stagnation_detected && result.stagnation_count >= 2) || result.duplicate_failed_ast)) requestFinalize('stagnation');
      if (requestWasFinalize && name === 'submit_program' && !result.eligible) {
        emergencySubmitFailedDemo = Boolean(
          result.repairable_submission_failure
          || (Array.isArray(result.feedback?.demos) && result.feedback.demos.some(d => d.kind !== 'pass'))
          || (Array.isArray(result.feedback?.queries) && result.feedback.queries.some(q => q.valid === false))
          || ['syntax', 'contract', 'runtime'].includes(result.feedback?.kind)
        );
      }
      state.tool_seconds += result.seconds ?? 0;
      if (result.is_error) state.tool_errors++;
      if (result.feedback?.workspace_reset) state.workspace_resets++;
      if (result.eligible) {
        state.status = 'eligible'; state.eligible = true; state.predictions = structuredClone(result.predictions);
        state.answer_source = routed ? 'overflow_verified_program' : 'submit_program';
        state.stop_reason = 'first_eligible_program';
        eligibleCandidates.push({ source: 'submit_program', wave, verified: true,
          demo_count: task.train?.length ?? 0, demo_pass_count: task.train?.length ?? 0,
          demo_diff_cells: 0, order: eligibleCandidates.length + 1,
          predictions: structuredClone(result.predictions) });
        savedEligibleResult = { predictions: structuredClone(state.predictions), answer_source: state.answer_source };
        if (name === 'submit_program' && !firstEligibleSubmissionSeen) {
          firstEligibleSubmissionSeen = true;
          const previous = previousPredictions?.map(p => p?.attempt_1 ?? p);
          const duplicate = Array.isArray(previous) && JSON.stringify(result.predictions) === JSON.stringify(previous);
          if (duplicate && duplicateExtendTokens > 0 && !duplicateExtendedOnce && outputLimit - state.output_tokens >= 2048) {
            duplicateExtendedOnce = true; state.duplicate_extended_once = true;
            outputLimit = Math.min(outputLimit, state.output_tokens + duplicateExtendTokens);
            state.output_limit = outputLimit; state.status = 'running'; state.stop_reason = null; setFinalizeState('EXPLORE');
            result = { ...result, feedback: { ...(typeof result.feedback === 'object' && result.feedback !== null
              ? result.feedback : { result: result.feedback }),
              duplicate_notice: 'This program passes every training example and matches a previously saved candidate. That candidate remains saved. If another general rule is supported by the examples, submit it; do not change outputs solely to make them different.' } };
          } else if (Array.isArray(result.feedback?.query_advisories) && result.feedback.query_advisories.length > 0
            && !duplicateExtendedOnce && !requestWasFinalize && state.final_checks < 3 && outputLimit - state.output_tokens >= 4096) {
            duplicateExtendedOnce = true; state.advisory_extended_once = true;
            state.output_limit = outputLimit; state.status = 'running'; state.stop_reason = null; setFinalizeState('EXPLORE');
            result = { ...result, feedback: { ...(typeof result.feedback === 'object' && result.feedback !== null
              ? result.feedback : { result: result.feedback }),
              advisory_notice: 'This program passes every training example and its predictions are saved. Review query_advisories on test_inputs: if the observed discrepancy is unintended (for example, an unhandled boundary or fallback leaving mask or default cells), submit a revised complete program; if it is justified by the rule, no change is needed.' } };
          }
        } else if (state.duplicate_extended_once && previousPredictions
          && JSON.stringify(result.predictions) === JSON.stringify(previousPredictions.map(p => p?.attempt_1 ?? p))) {
          state.status = 'running'; state.stop_reason = null;
        }
      } else if ((name === 'submit_program' || routed) && Array.isArray(result.candidate_predictions) && result.candidate_predictions.length > 0) {
        const passCount = result.demo_pass_count ?? 0;
        const diffCells = result.demo_diff_cells ?? 999999;
        const better = state.fallback_predictions.length === 0
          || passCount > state.fallback_demo_pass_count
          || (passCount === state.fallback_demo_pass_count && diffCells <= (state.fallback_demo_diff_cells ?? 999999));
        if (better) {
          state.fallback_predictions = result.candidate_predictions;
          state.fallback_demo_pass_count = passCount;
          state.fallback_demo_diff_cells = diffCells;
        }
      }
      if (name === 'python' && !routed && !result.eligible && pythonLimit > 12
        && (state.python_calls === 12 || state.python_calls > pythonNoticeAfter)) {
        const left = Math.max(0, pythonLimit - state.python_calls);
        const notice = state.python_calls === 12
          ? `Python experiment budget extended by ${pythonLimit - 12} additional calls (${pythonLimit} total). Remaining Python calls: ${left}.`
          : `Remaining Python calls: ${left}.`;
        result = { ...result, feedback: { ...(typeof result.feedback === 'object' && result.feedback !== null
          ? result.feedback : { result: result.feedback }), python_budget_notice: notice } };
      }
      if (lateBudgetNoticeTokens > 0 && !lateNoticeSent && finalizeState === 'EXPLORE'
        && !result.eligible && outputLimit - state.output_tokens <= lateBudgetNoticeTokens) {
        lateNoticeSent = true;
        result = { ...result, feedback: { ...(typeof result.feedback === 'object' && result.feedback !== null
          ? result.feedback : { result: result.feedback }),
          late_budget_notice: `Remaining output tokens: ${Math.max(0, outputLimit - state.output_tokens)}. Prepare your best complete program for submit_program.` } };
      }
      const content = [{ type: 'text', text: typeof result.feedback === 'string'
          ? result.feedback : JSON.stringify(result.feedback) }];
      return { content,
        details: result, isError: !!result.is_error, terminate: state.status === 'eligible' };
    }
  }));
  const streamFn = (m, context, options) => streamQwen(m, context, {
    ...options, apiKey: 'local', maxRetries: 0, reasoningEffort,
    timeoutMs: STREAM_TIMEOUT_MS,
    fetch: async (url, init) => {
      const requestUrl = typeof url === 'string' ? url : url?.url ?? String(url);
      // onPayload/emit can still fail: only the actual completion fetch crosses
      // the dispatch boundary, so a pending finalize cannot finish prematurely.
      if (requestUrl.includes('/chat/completions') && requestWasFinalize
        && ['PENDING', 'REPAIR_PENDING'].includes(finalizeState)) {
        setFinalizeState(finalizeState === 'REPAIR_PENDING' ? 'REPAIR_DISPATCHED' : 'DISPATCHED');
      }
      return longTimeoutFetch(url, init);
    },
    onResponse: async () => {
      if (activeStreamTiming) activeStreamTiming.headersAt = Date.now();
    },
    onProviderStreamEvent: async () => {
      if (activeStreamTiming) {
        const t = Date.now();
        activeStreamTiming.firstChunkAt ??= t;
        activeStreamTiming.lastChunkAt = t;
        activeStreamTiming.chunkCount++;
      }
    },
    onPayload: async payload => {
      let remaining = outputLimit - state.output_tokens;
      if (remaining <= 0) throw new Error('output_budget_exhausted');
      const requestSeed = state.seed_jitter_active ? seed + state.requests : seed;
      const request = { ...payload, messages: restoreReasoningOnlyPayload(payload.messages),
        temperature: 1, top_p: 0.95, top_k: 20, seed: requestSeed,
        tool_choice: 'auto', parallel_tool_calls: false,
        chat_template_kwargs: { enable_thinking: true, reasoning_effort: reasoningEffort, preserve_thinking: true },
        stream_options: { include_usage: true, continuous_usage_stats: true }, max_tokens: remaining };
      const applyFinalizeControls = () => {
        if (!['PENDING', 'REPAIR_PENDING'].includes(finalizeState)) return;
        request.tool_choice = { type: 'function', function: { name: 'submit_program' } };
        request.parallel_tool_calls = false;
        request.chat_template_kwargs = { enable_thinking: false, preserve_thinking: true };
        delete request.thinking_token_budget;
        const last = request.messages.at(-1);
        if (last?.role === 'tool') {
          const text = typeof last.content === 'string'
            ? last.content
            : Array.isArray(last.content)
              ? last.content.map(c => c?.text ?? '').join('\n')
              : String(last.content ?? '');
          if (!text.endsWith(FINALIZE_PROMPT)
            && text !== 'This tool budget is exhausted. Submit your complete program.') {
            const updated = typeof last.content === 'string'
              ? `${last.content}\n\n${FINALIZE_PROMPT}`
              : Array.isArray(last.content)
                ? [...last.content, { type: 'text', text: `\n\n${FINALIZE_PROMPT}` }]
                : `${text}\n\n${FINALIZE_PROMPT}`;
            request.messages = [...request.messages.slice(0, -1), { ...last, content: updated }];
          }
        }
      };
      if (finalizeReserve > 0 && canFinalize() && remaining <= finalizeReserve + 512) {
        requestFinalize('reserved_output', false, true);
      }
      applyFinalizeControls();
      let encoded = { count: 0, tokens: null };
      // Tokenizer-only HTTP call: preserves complete history, caps output to the
      // remaining context instead of pruning reasoning or silently truncating it.
      const encodeRequest = async () => {
        const response = await longTimeoutFetch(m.baseUrl.replace(/\/v1\/?$/, '') + '/tokenize', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
          body: JSON.stringify({ model: m.id, messages: request.messages, tools: request.tools,
            add_generation_prompt: true, tool_choice: request.tool_choice, parallel_tool_calls: request.parallel_tool_calls,
            chat_template_kwargs: request.chat_template_kwargs }) });
        if (!response.ok) throw new Error(`tokenize_http_${response.status}`);
        return response.json();
      };
      if (tokenize) {
        encoded = await encodeRequest();
        // Context pressure is only known after tokenization. When the previous
        // turn lacked a tool call, add the same explicit finalize instruction
        // used for output pressure and account for its actual prompt tokens.
        if (finalizeReserve > 0 && canFinalize() && m.contextWindow - encoded.count <= finalizeReserve + 512) {
          requestFinalize('reserved_context', false, true);
          applyFinalizeControls();
          if (lastTurnHadTools === false) {
            request.messages = [...request.messages, { role: 'user', content: FINALIZE_PROMPT }];
          }
          encoded = await encodeRequest();
        }
        if (m.contextWindow - encoded.count <= 0) {
          state.stop_reason = 'context_budget_exhausted';
          throw new Error(state.stop_reason);
        }
      }
      if (scheduler) {
        if (!tokenize) throw new Error('KV admission needs the actual prompt token count');
        state.phase = 'waiting_for_kv';
        const tokens = reservationTokens(encoded.count, remaining, m.contextWindow);
        await emit({ type: 'kv_wait', at: Date.now(), reservation_tokens: tokens }, state);
        const admission = await scheduler.acquire(taskId, tokens);
        if (signal?.aborted) throw new Error('admission_aborted');
        state.admission_wait_seconds += admission.wait_ms / 1000;
        request.priority = admission.continuation ? -1 : 0;
        if (governor && !admission.continuation && !state.governor_decision) {
          // Time-aware cap is fixed once, at the task's cold admission.
          const decision = governor.allocate ? governor.allocate(taskId, { wave }) : governor.admit(taskId);
          if (decision.admit === false) throw new Error(decision.reason ?? 'wave2_budget_insufficient');
          governor.startTask?.(taskId, decision.limit, { wave });
          state.governor_decision = { limit: decision.limit, reduced: decision.reduced,
            per_task_tokens: decision.per_task_tokens, tps: decision.tps, unstarted: decision.unstarted };
          outputLimit = Math.min(outputLimit, decision.limit);
          state.output_limit = outputLimit; state.exploration_output_limit = outputLimit;
          remaining = outputLimit - state.output_tokens;
          await emit({ type: 'output_cap', at: Date.now(), ...state.governor_decision }, state);
        }
        state.phase = 'admitted';
        await emit({ type: 'kv_admitted', at: Date.now(), reservation_tokens: tokens, ...admission }, state);
      }
      // A legacy cold-admission governor may reduce the cap after the first
      // tokenize. Apply any resulting reserve transition before the final encode.
      if (finalizeReserve > 0 && canFinalize() && remaining <= finalizeReserve + 512) {
        requestFinalize('reserved_admission', false, true); applyFinalizeControls();
        if (tokenize) encoded = await encodeRequest();
      }
      const cachePlan = prefix.prepare(encoded);
      requestPromptTokens = encoded.count;
      const contextRemaining = m.contextWindow - encoded.count;
      requestWasFinalize = ['PENDING', 'REPAIR_PENDING'].includes(finalizeState);
      requestReservedFinalize = !requestWasFinalize && finalizeReserve > 0 && canFinalize();
      if (requestWasFinalize) {
        request.max_tokens = Math.min(remaining, contextRemaining, currentFinalizeAllowance);
        requestLimitReason = 'finalize_allowance';
      } else {
        // Opt-in reserved finalization retains its pre-existing exploration cap.
        const outputAllowance = remaining - (requestReservedFinalize ? finalizeReserve : 0);
        const contextAllowance = contextRemaining - (requestReservedFinalize ? finalizeReserve + 512 : 0);
        request.max_tokens = Math.min(outputAllowance, contextAllowance);
        requestLimitReason = outputAllowance <= contextAllowance ? 'output' : 'context';
        const thinkCap = state.requests === 0 ? turn1ThinkCap : turnThinkCap;
        if (thinkCap > 0) request.thinking_token_budget = Math.min(thinkCap,
          Math.max(256, request.max_tokens - codeReserve));
      }
      if (finalizeState === 'REPAIR_PENDING' && request.max_tokens < 1536) {
        state.status = 'censored'; state.stop_reason = 'repair_context_or_budget_insufficient';
        setFinalizeState('DONE'); throw new Error(state.stop_reason);
      }
      if (request.max_tokens <= 0) throw new Error('output_or_context_limit');
      requestOutputAllowance = request.max_tokens;
      state.phase = 'generating';
      state.first_dispatch_at ??= Date.now() / 1000;
      state.requests++; state.last_request_seed = requestSeed; state.last_request_limit_reason = requestLimitReason;
      if (requestWasFinalize) {
        emergencySubmitFailedDemo = false; emergencySubmissionsThisRequest = 0;
      }
      activeStreamTiming = { dispatchedAt: Date.now(), headersAt: null, firstChunkAt: null, lastChunkAt: null, chunkCount: 0 };
      await emit({ type: 'request', request_number: state.requests, at: Date.now(),
        cache_plan: cachePlan, payload: request }, state);
      return request;
    }
  });
  const config = { model, reasoning: reasoningEffort, maxTokens: outputLimit, toolExecution: 'sequential',
    convertToLlm: preserveReasoningOnlyMessages,
    beforeToolCall: ({ assistantMessage, toolCall }) => {
      settle(assistantMessage);
      const exhausted = state.final_checks >= 3;
      if (state.status !== 'running' || exhausted) return { block: true,
        reason: state.status !== 'running' ? 'Task finished; no further action.' : 'This tool budget is exhausted. Submit your complete program.',
        terminate: state.status !== 'running' || state.final_checks >= 3 };
      if (toolCall.name === 'python' && state.overflow_probes_attempted >= 1) return { block: true,
        reason: 'Python budget exhausted. Call submit_program with your complete transform(grid) program.' };
      if (['PENDING', 'REPAIR_PENDING'].includes(finalizeState)) return { block: true,
        reason: 'Finalization is scheduled. Call submit_program with your complete program on the next turn.' };
      if (requestWasFinalize && emergencySubmissionsThisRequest >= 1) return { block: true,
        reason: 'Only one submit_program is allowed per finalization turn.' };
      const finalCheck = toolCall.name === 'submit_program' || (toolCall.name === 'python' && state.python_calls >= pythonLimit);
      if (finalizeReserve > 0 && canFinalize() && finalCheck && state.final_checks >= 2) {
        return { block: true, reason: 'The last final submission is reserved. Call submit_program with your best complete program on the final turn.' };
      }
      if (requestWasFinalize && toolCall.name !== 'submit_program') {
        return { block: true, reason: 'Call submit_program with your best complete program now.' };
      }
    },
    finishTurn: ({ message }) => {
      settle(message);
      const hasTools = message.content.some(c => c.type === 'toolCall');
      lastTurnHadTools = hasTools;
      if (state.status !== 'running') return { action: 'end' };
      if (message.stopReason === 'error' || message.stopReason === 'aborted') {
        state.status = message.stopReason === 'aborted' ? 'deadline' : 'task_error';
        state.stop_reason ??= message.errorMessage ?? message.stopReason;
        if (isBackendFailure(state.stop_reason)) onBackendFailure(state.stop_reason);
        return { action: 'end' };
      }
      if (state.unknown_usage_requests) {
        state.status = 'task_error'; state.stop_reason = 'missing_usage_receipt'; return { action: 'end' };
      }
      // A trigger raised during tool execution schedules a *later* request.
      // Never end that same turn merely because finalization is pending.
      if (['PENDING', 'REPAIR_PENDING'].includes(finalizeState)) {
        pendingFinalizePrompt ||= !hasTools; return { action: 'continue' };
      }
      if (['DISPATCHED', 'REPAIR_DISPATCHED'].includes(finalizeState)) {
        const contextLeft = model.contextWindow - requestPromptTokens - message.usage.output;
        if (finalizeState === 'DISPATCHED' && contextLeft >= 1536 && requestRepair()) return { action: 'continue' };
        setFinalizeState('DONE'); state.stop_reason = 'finalize_completed'; return { action: 'end' };
      }
      if (finalizeReserve > 0 && canFinalize() && outputLimit - state.output_tokens > 0
        && ((message.stopReason === 'length' && requestReservedFinalize)
          || outputLimit - state.output_tokens <= finalizeReserve + 512)) {
        if (requestFinalize('reserved_budget', !hasTools, true)) return { action: 'continue' };
      }
      if (state.output_tokens >= outputLimit || message.stopReason === 'length') {
        const contextLeft = model.contextWindow - requestPromptTokens - message.usage.output;
        if (contextLeft > 2560 && (state.output_tokens >= outputLimit || (requestLimitReason === 'output' && message.usage.output >= requestOutputAllowance))
          && requestFinalize('output_budget', !hasTools)) return { action: 'continue' };
        state.stop_reason = 'output_or_context_limit'; return { action: 'end' };
      }
      if (state.final_checks >= 3) { state.stop_reason = 'final_checks_exhausted'; return { action: 'end' }; }
      if (!hasTools) {
        if (savedEligibleResult && state.advisory_extended_once && !state.duplicate_extended_once) {
          state.stop_reason = 'advisory_review_accepted'; return { action: 'end' };
        }
        if (corrections++ >= 2) { state.stop_reason = 'no_native_tool_call'; return { action: 'end' }; }
        pendingCorrection = true; return { action: 'continue' };
      }
    },
    prepareNextTurn: () => {
      if (pendingFinalizePrompt) {
        pendingFinalizePrompt = false;
        return { messages: [{ role: 'user', timestamp: Date.now(),
          content: FINALIZE_PROMPT }] };
      }
      if (!pendingCorrection) return;
      pendingCorrection = false;
      return { messages: [{ role: 'user', timestamp: Date.now(),
        content: 'Continue by calling python or submit_program through the function interface.' }] };
    }
  };
  try {
    await runAgentLoop([{ role: 'user', content: buildInitialUserContent(task), timestamp: Date.now() }],
      { messages: [{ role: 'system', content: rewriteSystemPythonLimit
        ? SYSTEM.replace('at most 12 Python', `at most ${pythonLimit} Python`) : SYSTEM, timestamp: Date.now() }], tools }, config,
      async event => {
        if (event.type === 'message_update' && event.message?.usage?.output && state.requests !== accountedRequests) {
          const generated = event.message.usage.output;
          liveUsage(state.output_tokens + generated);
          scheduler?.update(taskId, reservationTokens(requestPromptTokens + generated,
            requestOutputAllowance - generated, model.contextWindow), { prefilled: true });
        }
        if (event.type === 'message_end') settle(event.message);
        if (event.type === 'tool_execution_start') state.phase = 'tool';
        for (const receipt of receipts.splice(0)) await emit(receipt, state);
        for (const diag of streamDiagnostics.splice(0)) await emit(diag, state);
        // The provider's partial message includes continuous usage and complete
        // reasoning; save a separate live snapshot periodically in the caller.
        await emit(event, state);
      }, signal, streamFn);
  } catch (error) {
    state.stop_reason = String(error); state.status = signal?.aborted ? 'deadline' : 'task_error';
    if (isBackendFailure(error)) onBackendFailure(error);
  } finally {
    for (const receipt of receipts.splice(0)) await emit(receipt, state);
    for (const diag of streamDiagnostics.splice(0)) await emit(diag, state);
    // Whole-task reservation spans every Pi tool turn. No GPU request or lease
    // is held by an already accepted, failed, or deadline-cancelled task.
    scheduler?.release(taskId);
    if (governor?.finishTask) governor.finishTask(taskId, state.output_tokens, { wave });
    else governor?.finish(taskId, state.output_tokens);
    try { await bridge.close(); }
    catch (error) {
      state.status = signal?.aborted ? 'deadline' : 'task_error';
      state.stop_reason = `workspace_close_failed: ${String(error)}`;
      state.tool_errors++;
    }
    rememberVerified(bridge.verifiedSnapshot);
  }
  if (savedEligibleResult) {
    if (state.status !== 'eligible') state.extension_stop_reason = state.stop_reason;
    state.status = 'eligible'; state.eligible = true;
    state.predictions = savedEligibleResult.predictions; state.answer_source = savedEligibleResult.answer_source;
  }
  if (state.status === 'running') state.status = signal?.aborted ? 'deadline' : 'censored';
  if (signal?.reason === 'engine_failure' && !['eligible', 'censored'].includes(state.status)) {
    state.status = state.requests > 0 ? 'engine_interrupted' : 'not_dispatched';
    state.stop_reason = 'shared_engine_failure';
  }
  if (!state.predictions.length && state.verified_fallback_predictions.length) {
    state.predictions = structuredClone(state.verified_fallback_predictions);
    state.verified_fallback_used = true;
    state.answer_source = 'verified_fallback';
    // Preserve the generation stop/error status independently of the saved answer.
  }
  // Silent pool -> per-query attempt_1 / distinct attempt_2 (no extra generation).
  let pool = [];
  try { pool = bridge.poolCandidates ?? []; } catch { pool = []; }
  pool = [...pool.map(c => ({ ...c, wave })), ...eligibleCandidates];
  if (state.verified_fallback_predictions.length) pool.push({ source: 'verify_demos', verified: true, wave,
    action_id: state.verified_fallback_action_id, demo_count: task.train?.length ?? 0,
    demo_pass_count: task.train?.length ?? 0, demo_diff_cells: 0,
    predictions: structuredClone(state.verified_fallback_predictions) });
  state.candidate_pool = pool; state.pool_candidates = pool;
  state.attempts = selectAttempts({ queryCount: task.test?.length ?? 0, wave, primary: state.predictions,
    fallback: state.fallback_predictions, pool });
  state.distinct_attempt_2 = state.attempts.filter(a => a.attempt_2_source?.startsWith('pool:')).length;
  state.seconds = (Date.now() - started) / 1000;
  state.service_seconds = state.first_dispatch_at === null ? 0 : Date.now() / 1000 - state.first_dispatch_at;
  if (finalizeState !== 'EXPLORE') setFinalizeState('DONE');
  state.phase = 'finished';
  return state;
}
