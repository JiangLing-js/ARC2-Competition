// A lease retains the current conversation plus one short growth window.
// The output allowance is 65K; vLLM owns paging of the live requests.
//
// Cold admission is shaped to avoid synchronized cohorts (2026-10-24 L4 audit):
// - startup ramp: `initialTasks` may enter immediately, then the resident
//   ceiling grows by one every `rampIntervalMs` up to `maxTasks`;
// - after the initial burst, at most one cold admission per `minColdIntervalMs`;
// - narrow KV hysteresis (pause 0.84 / resume 0.78) so cold admission is not
//   starved for an hour and then released as a 20-task burst.
//
// Two-layer KV anti-overflow protection (V26 4xL4 forensic fix):
// - Layer 1 (Working-Set & Tool-Gap Protection): In vLLM V1, tasks inside
//   `run_python` between turns have `ref_cnt = 0` on their KV blocks and
//   disappear from `vllm:kv_cache_usage_perc`. Multi-turn leases therefore
//   enforce a committed working-set floor (including forward maturity reserve
//   for deep 65K trajectories) so tool-gap telemetry dips never trigger false
//   cold admissions.
// - Layer 2 (Deepest-First Continuation Backpressure): When active KV pressure
//   is severe (`kvUsage >= 0.86`, `waiting > 0`, `preempted`, or actual
//   multi-turn reservations exceed budget), the deepest leases closest to 65K
//   completion always proceed with `wait_ms = 0`, while the shallowest tail
//   leases pause briefly until a deep lease finishes or pressure clears.
export class KvScheduler {
  constructor({ capacityTokens, maxTasks = 24, utilization = 0.88,
    pauseAt = 0.84, resumeAt = 0.78,
    initialTasks = 20, rampIntervalMs = 20_000, minColdIntervalMs = 8_000,
    telemetryMaxAgeMs = 15_000, observationLagMs = 10_000,
    coldGrowthWindowTokens = 16384, feedbackReserve = 4096,
    maturityThresholdTokens = 28_000, maturityFloorTokens = 52_000, maturityMaxTokens = 68_000,
    continuationPauseAt = 0.86, continuationSevereAt = 0.90,
    continuationActiveUtilization = 0.84, continuationSevereUtilization = 0.80,
    continuationMaxWaitMs = 120_000, continuationEscapeIntervalMs = 15_000,
    signal, now = Date.now, setTimer = setTimeout, clearTimer = clearTimeout }) {
    if (!(capacityTokens > 0)) throw new Error('KV capacity must come from the server startup log');
    if (!(resumeAt < pauseAt)) throw new Error('resumeAt must be below pauseAt');
    if (!(Number.isInteger(initialTasks) && initialTasks >= 1)) throw new Error('initialTasks must be a positive integer');
    if (!(Number.isFinite(telemetryMaxAgeMs) && telemetryMaxAgeMs > 0)) throw new Error('telemetryMaxAgeMs must be positive and finite');
    if (!(Number.isFinite(observationLagMs) && observationLagMs >= 0)) throw new Error('observationLagMs must be non-negative and finite');
    this.capacityTokens = capacityTokens;
    this.budget = Math.floor(capacityTokens * utilization);
    this.maxTasks = maxTasks;
    this.pauseAt = pauseAt; this.resumeAt = resumeAt;
    this.initialTasks = Math.min(initialTasks, maxTasks);
    this.rampIntervalMs = rampIntervalMs; this.minColdIntervalMs = minColdIntervalMs;
    this.telemetryMaxAgeMs = telemetryMaxAgeMs; this.observationLagMs = observationLagMs;
    this.coldGrowthWindowTokens = coldGrowthWindowTokens; this.feedbackReserve = feedbackReserve;
    this.maturityThresholdTokens = maturityThresholdTokens;
    this.maturityFloorTokens = maturityFloorTokens;
    this.maturityMaxTokens = maturityMaxTokens;
    this.continuationPauseAt = continuationPauseAt;
    this.continuationSevereAt = continuationSevereAt;
    this.continuationActiveUtilization = continuationActiveUtilization;
    this.continuationSevereUtilization = continuationSevereUtilization;
    this.continuationMaxWaitMs = continuationMaxWaitMs;
    this.continuationEscapeIntervalMs = continuationEscapeIntervalMs;
    this.now = now; this.signal = signal;
    this.setTimer = setTimer; this.clearTimer = clearTimer;
    this.leases = new Map(); this.queue = []; this.continuationQueue = [];
    this.kvWatermarkPaused = false;
    this.pressurePaused = false; this.telemetry = null;
    this.preemptions = null;
    this.firstColdAt = null; this.lastColdAt = null; this.coldAdmissions = 0;
    this.lastContinuationEscapeAt = null;
    this.wakeTimer = null; this.wakeAt = null; this.lastBlock = null;
    this.abortHandler = () => {
      this.clearWake();
      for (const item of this.continuationQueue.splice(0)) item.reject(new Error('admission_aborted'));
      for (const item of this.queue.splice(0)) item.reject(new Error('admission_aborted'));
    };
    signal?.addEventListener('abort', this.abortHandler, { once: true });
  }

  get reservedTokens() {
    return [...this.leases.values()].reduce((sum, lease) => sum + lease.tokens, 0);
  }

  isMultiTurnLease(lease) {
    return (lease.continuations ?? 0) > 0 || (lease.peakTokens ?? lease.tokens) > (lease.initialTokens ?? lease.tokens);
  }

  leaseWorkingSetTokens(lease) {
    const base = Math.max(lease.tokens, lease.peakTokens ?? lease.tokens);
    if (!this.isMultiTurnLease(lease)) return lease.tokens;
    if (this.budget >= 500_000 && (lease.continuations ?? 0) >= 2 && base >= this.maturityThresholdTokens) {
      const speculativeFloor = Math.min(this.maturityMaxTokens, this.maturityFloorTokens);
      return Math.max(base, speculativeFloor);
    }
    return base;
  }

  isSevereContinuationPressure(at = this.now()) {
    if (!this.isTelemetryFresh(at)) return false;
    return Boolean(this.telemetry.kvUsage >= this.continuationSevereAt
      || this.telemetry.waiting > 0
      || this.telemetry.preempted);
  }

  committedWorkingSetTokens() {
    let multiTurnCount = 0, total = 0;
    for (const lease of this.leases.values()) {
      if (this.isMultiTurnLease(lease)) multiTurnCount++;
      total += this.leaseWorkingSetTokens(lease);
    }
    return multiTurnCount > 0 ? total : 0;
  }

  actualWorkingSetTokens() {
    let multiTurnCount = 0, total = 0;
    for (const lease of this.leases.values()) {
      if (this.isMultiTurnLease(lease)) multiTurnCount++;
      total += Math.max(lease.tokens, lease.peakTokens ?? lease.tokens);
    }
    return multiTurnCount > 0 ? total : 0;
  }

  activeContinuationCeiling(at = this.now()) {
    if (this.leases.size <= 1) return Infinity;
    const fresh = this.isTelemetryFresh(at);
    const preempted = Boolean(fresh && this.telemetry?.preempted);
    const severeTelemetry = fresh && (this.telemetry.kvUsage >= this.continuationSevereAt || this.telemetry.waiting > 0 || preempted);
    if (severeTelemetry) return Math.floor(this.capacityTokens * this.continuationSevereUtilization);
    const highTelemetry = fresh && this.telemetry.kvUsage >= this.continuationPauseAt;
    if (highTelemetry) return Math.floor(this.capacityTokens * this.continuationActiveUtilization);
    if (this.actualWorkingSetTokens() > this.budget) return this.budget;
    return Infinity;
  }

  isAllowedContinuation(taskId, at = this.now()) {
    const ceiling = this.activeContinuationCeiling(at);
    if (!Number.isFinite(ceiling)) return true;
    const sorted = [...this.leases.entries()].sort(([, a], [, b]) => {
      const tokA = Math.max(a.tokens, a.peakTokens ?? a.tokens);
      const tokB = Math.max(b.tokens, b.peakTokens ?? b.tokens);
      if (tokB !== tokA) return tokB - tokA;
      return (a.admittedAt - b.admittedAt) || ((a.admittedSeq ?? 0) - (b.admittedSeq ?? 0));
    });
    let cumulative = 0;
    for (let rank = 0; rank < sorted.length; rank++) {
      const [id, lease] = sorted[rank];
      const tok = Math.max(lease.tokens, lease.peakTokens ?? lease.tokens);
      cumulative += tok;
      const allowed = rank === 0 || cumulative <= ceiling;
      if (id === taskId) return allowed;
    }
    return true;
  }

  isTelemetryFresh(at = this.now()) {
    return this.telemetry !== null && at >= this.telemetry.at
      && at - this.telemetry.at <= this.telemetryMaxAgeMs;
  }

  expireStalePressure(at = this.now()) {
    if (this.telemetry !== null && !this.isTelemetryFresh(at)) {
      this.kvWatermarkPaused = false;
      this.pressurePaused = false;
    }
  }

  occupancy(at = this.now()) {
    const physicalTokens = (this.telemetry?.kvUsage ?? 0) * this.capacityTokens;
    const workingSetTokens = this.committedWorkingSetTokens();
    const fresh = this.isTelemetryFresh(at);
    if (!fresh) return { mode: this.telemetry ? 'stale_reservations' : 'reservations',
      physical_tokens: physicalTokens, unobserved_cold_tokens: 0, recent_growth_tokens: 0,
      working_set_tokens: workingSetTokens,
      occupied_tokens: Math.max(this.reservedTokens, workingSetTokens) };
    // Fresh physical occupancy already includes shared prefixes and existing
    // contexts. Adding every lease's 16K growth window again starves admission.
    // Keep full cold reservations until the first generated token acknowledges
    // that prefill completed, then until a newer metric passes the lag guard.
    // Elapsed time alone does not prove a long chunked prefill is represented
    // in physical occupancy. Missing token usage remains conservative.
    // For multi-turn leases, `workingSetTokens` prevents transient `run_python`
    // tool-gap dips (`ref_cnt = 0` in vLLM) from under-reporting committed KV.
    const observedBefore = this.telemetry.at - this.observationLagMs;
    let pending = 0, growth = 0;
    for (const lease of this.leases.values()) {
      if (lease.prefilledAt === null || lease.prefilledAt >= observedBefore) pending += lease.tokens;
      else for (const change of lease.growth) {
        if (change.at >= observedBefore) growth += change.tokens;
      }
    }
    return { mode: 'physical_with_pending', physical_tokens: physicalTokens,
      unobserved_cold_tokens: pending, recent_growth_tokens: growth,
      working_set_tokens: workingSetTokens,
      occupied_tokens: Math.max(physicalTokens + pending + growth, workingSetTokens) };
  }

  // Resident-task ceiling during the startup ramp.
  residentCeiling(at = this.now()) {
    if (this.firstColdAt === null || !(this.rampIntervalMs > 0)) {
      return this.firstColdAt === null ? this.initialTasks : this.maxTasks;
    }
    const steps = Math.floor((at - this.firstColdAt) / this.rampIntervalMs);
    return Math.min(this.maxTasks, this.initialTasks + Math.max(0, steps));
  }

  // Earliest time the cold-admission rate limit allows another task.
  nextColdAllowedAt() {
    if (this.coldAdmissions < this.initialTasks || this.lastColdAt === null || !(this.minColdIntervalMs > 0)) return -Infinity;
    return this.lastColdAt + this.minColdIntervalMs;
  }

  update(taskId, tokens, { prefilled = false } = {}) {
    const lease = this.leases.get(taskId);
    if (lease) {
      const at = this.now(), delta = Math.max(0, tokens - lease.tokens);
      if (prefilled) lease.prefilledAt ??= at;
      lease.growth = lease.growth.filter(x => x.at >= at - this.telemetryMaxAgeMs - this.observationLagMs);
      if (delta) lease.growth.push({ at, tokens: delta });
      lease.tokens = tokens;
      lease.peakTokens = Math.max(lease.peakTokens ?? tokens, tokens);
    }
  }

  async acquire(taskId, tokens) {
    if (this.signal?.aborted) throw new Error('admission_aborted');
    // Warm continuations in the deepest active prefix proceed immediately so
    // tasks closest to 65K completion finish and free KV capacity first; when
    // KV pressure is severe, shallower tail leases wait briefly behind them.
    if (this.leases.has(taskId)) {
      const at = this.now();
      const lease = this.leases.get(taskId);
      lease.continuations = (lease.continuations ?? 0) + 1;
      this.update(taskId, tokens);
      this.expireStalePressure(at);
      if (this.isAllowedContinuation(taskId, at)) {
        return { continuation: true, wait_ms: 0 };
      }
      return new Promise((resolve, reject) => {
        this.continuationQueue.push({ taskId, enqueued: at, resolve, reject });
        this.pump();
      });
    }
    if (tokens > this.budget) throw new Error(`task_KV_reservation_${tokens}_exceeds_capacity_${this.budget}`);
    return new Promise((resolve, reject) => {
      this.queue.push({ taskId, tokens, enqueued: this.now(), resolve, reject });
      this.pump();
    });
  }

  clearWake() {
    if (this.wakeTimer !== null) this.clearTimer(this.wakeTimer);
    this.wakeTimer = null; this.wakeAt = null;
  }

  scheduleWake(at) {
    if (this.signal?.aborted || !Number.isFinite(at)) return;
    if (this.wakeTimer !== null && this.wakeAt <= at) return;
    this.clearWake();
    this.wakeAt = at;
    this.wakeTimer = this.setTimer(() => { this.wakeTimer = null; this.wakeAt = null; this.pump(); },
      Math.max(0, at - this.now()) + 5);
    this.wakeTimer?.unref?.();
  }

  pump() {
    if (this.signal?.aborted) return;
    const nowAt = this.now();
    this.expireStalePressure(nowAt);
    this.lastBlock = null;
    if (this.continuationQueue.length) {
      this.continuationQueue.sort((a, b) => {
        const leaseA = this.leases.get(a.taskId);
        const leaseB = this.leases.get(b.taskId);
        const tokA = leaseA ? Math.max(leaseA.tokens, leaseA.peakTokens ?? leaseA.tokens) : 0;
        const tokB = leaseB ? Math.max(leaseB.tokens, leaseB.peakTokens ?? leaseB.tokens) : 0;
        if (tokB !== tokA) return tokB - tokA;
        if (leaseA && leaseB) {
          const admDiff = (leaseA.admittedAt - leaseB.admittedAt) || ((leaseA.admittedSeq ?? 0) - (leaseB.admittedSeq ?? 0));
          if (admDiff !== 0) return admDiff;
        }
        return a.enqueued - b.enqueued;
      });
      for (let i = 0; i < this.continuationQueue.length; ) {
        const item = this.continuationQueue[i];
        const at = this.now();
        const waited = at - item.enqueued;
        const allowedByPrefix = !this.leases.has(item.taskId) || this.isAllowedContinuation(item.taskId, at);
        const canEscapeTimeout = !allowedByPrefix
          && waited >= this.continuationMaxWaitMs
          && !this.isSevereContinuationPressure(at)
          && (this.lastContinuationEscapeAt === null || at - this.lastContinuationEscapeAt >= this.continuationEscapeIntervalMs);
        if (allowedByPrefix || canEscapeTimeout) {
          if (canEscapeTimeout) this.lastContinuationEscapeAt = at;
          this.continuationQueue.splice(i, 1);
          item.resolve({ continuation: true, wait_ms: Math.max(0, waited) });
        } else {
          i++;
        }
      }
      if (this.continuationQueue.length) {
        const escapeReadyAt = this.lastContinuationEscapeAt === null
          ? -Infinity
          : this.lastContinuationEscapeAt + this.continuationEscapeIntervalMs;
        const earliestTimeout = this.isSevereContinuationPressure(nowAt)
          ? Infinity
          : Math.min(...this.continuationQueue.map(item => Math.max(item.enqueued + this.continuationMaxWaitMs, escapeReadyAt)));
        const staleAt = this.isTelemetryFresh(nowAt) ? this.telemetry.at + this.telemetryMaxAgeMs + 1 : Infinity;
        this.scheduleWake(Math.min(earliestTimeout, staleAt));
        this.lastBlock = 'continuation_backpressure';
        return;
      }
    }
    if (this.pressurePaused) {
      this.lastBlock = 'kv_pressure';
      if (this.telemetry && this.queue.length) {
        this.scheduleWake(this.telemetry.at + this.telemetryMaxAgeMs + 1);
      }
      return;
    }
    while (this.queue.length) {
      const at = this.now();
      if (this.leases.size >= this.maxTasks) { this.lastBlock = 'max_tasks'; break; }
      const ceiling = this.residentCeiling(at);
      if (this.leases.size >= ceiling) {
        this.lastBlock = 'startup_ramp';
        this.scheduleWake(this.firstColdAt + (ceiling - this.initialTasks + 1) * this.rampIntervalMs);
        break;
      }
      const allowedAt = this.nextColdAllowedAt();
      if (at < allowedAt) { this.lastBlock = 'cold_rate_limit'; this.scheduleWake(allowedAt); break; }
      const next = this.queue[0];
      const occupied = this.occupancy(at).occupied_tokens;
      if (occupied + next.tokens > this.budget) {
        this.lastBlock = 'kv_budget';
        if (this.isTelemetryFresh(at)) {
          this.scheduleWake(this.telemetry.at + this.telemetryMaxAgeMs + 1);
        }
        break;
      }
      this.queue.shift();
      this.leases.set(next.taskId, {
        tokens: next.tokens,
        initialTokens: next.tokens,
        peakTokens: next.tokens,
        continuations: 0,
        admittedAt: at,
        admittedSeq: this.coldAdmissions,
        prefilledAt: null,
        growth: [],
      });
      this.firstColdAt ??= at;
      this.lastColdAt = at; this.coldAdmissions++;
      next.resolve({ continuation: false, wait_ms: at - next.enqueued });
    }
  }

  observe(metrics, { sampledAt = this.now() } = {}) {
    const receivedAt = this.now();
    if (!(Number.isFinite(sampledAt) && sampledAt <= receivedAt)
      || !(Number.isFinite(metrics.kvUsage) && metrics.kvUsage >= 0 && metrics.kvUsage <= 1)
      || !(Number.isFinite(metrics.waiting) && metrics.waiting >= 0)
      || !(Number.isFinite(metrics.preemptions) && metrics.preemptions >= 0)) {
      throw new Error('Invalid KV telemetry');
    }
    if (this.telemetry && sampledAt < this.telemetry.at) return;
    const preempted = this.preemptions !== null && metrics.preemptions > this.preemptions;
    this.preemptions = metrics.preemptions;
    this.telemetry = { ...metrics, preempted, at: sampledAt, receivedAt };
    const workingSetUsage = this.committedWorkingSetTokens() / this.capacityTokens;
    if (metrics.kvUsage >= this.pauseAt || preempted) {
      this.kvWatermarkPaused = true;
    } else if (Math.max(metrics.kvUsage, workingSetUsage) <= this.resumeAt && metrics.waiting === 0) {
      this.kvWatermarkPaused = false;
    }
    this.pressurePaused = Boolean(this.kvWatermarkPaused || metrics.waiting > 0);
    this.pump();
  }

  release(taskId) {
    this.leases.delete(taskId);
    for (let i = this.continuationQueue.length - 1; i >= 0; i--) {
      if (this.continuationQueue[i].taskId === taskId) {
        const [item] = this.continuationQueue.splice(i, 1);
        item.resolve({ continuation: true, wait_ms: Math.max(0, this.now() - item.enqueued) });
      }
    }
    this.pump();
  }

  snapshot() {
    const at = this.now();
    this.expireStalePressure(at);
    return { policy: 'physical_kv_with_pending_cold_ramped', growth_window_tokens: this.coldGrowthWindowTokens,
      pause_at: this.pauseAt, resume_at: this.resumeAt,
      max_tasks: this.maxTasks, initial_tasks: this.initialTasks,
      ramp_interval_ms: this.rampIntervalMs, min_cold_interval_ms: this.minColdIntervalMs,
      resident_ceiling: this.residentCeiling(at), cold_admissions: this.coldAdmissions,
      last_block_reason: this.lastBlock,
      resident_tasks: [...this.leases.keys()],
      awaiting_first_output: [...this.leases].filter(([, lease]) => lease.prefilledAt === null).map(([id]) => id),
      backpressured_continuations: this.continuationQueue.map(item => item.taskId),
      queued_tasks: this.queue.map(item => item.taskId),
      capacity_tokens: this.capacityTokens, admission_budget_tokens: this.budget,
      telemetry_max_age_ms: this.telemetryMaxAgeMs, observation_lag_ms: this.observationLagMs,
      occupancy: this.occupancy(at),
      reserved_tokens: this.reservedTokens, cold_admission_paused: this.pressurePaused,
      kv_watermark_paused: this.kvWatermarkPaused,
      telemetry: this.telemetry };
  }

  close() {
    this.signal?.removeEventListener('abort', this.abortHandler);
    this.clearWake();
    for (const item of this.continuationQueue.splice(0)) item.reject(new Error('admission_closed'));
    for (const item of this.queue.splice(0)) item.reject(new Error('admission_closed'));
  }
}

export function reservationTokens(promptTokens, remainingOutput, contextWindow, feedbackReserve = 4096, coldGrowthWindowTokens = 16384) {
  return Math.min(contextWindow, promptTokens + Math.min(Math.max(0, remainingOutput), coldGrowthWindowTokens) + feedbackReserve);
}

export function capacityFromLog(text) {
  const matches = [...text.matchAll(/GPU KV cache size:\s*([\d,]+)\s*tokens/g)];
  if (!matches.length) throw new Error('GPU KV cache size is absent from the server startup log');
  // TP ranks share one logical context budget. Never sum a per-engine number.
  return Math.min(...matches.map(match => Number(match[1].replaceAll(',', ''))));
}

export function parseMetrics(text) {
  const values = new Map();
  for (const line of text.split('\n')) {
    const match = line.match(/^(vllm:[\w:]+)(?:\{.*\})?\s+([\d.eE+-]+)$/);
    if (match) {
      const list = values.get(match[1]) ?? [];
      list.push(Number(match[2])); values.set(match[1], list);
    }
  }
  const max = name => Math.max(0, ...(values.get(name) ?? []));
  const sum = name => (values.get(name) ?? []).reduce((a, b) => a + b, 0);
  if (!values.has('vllm:kv_cache_usage_perc')) throw new Error('KV metric is missing');
  return { kvUsage: max('vllm:kv_cache_usage_perc'), waiting: sum('vllm:num_requests_waiting'),
    running: sum('vllm:num_requests_running'), preemptions: sum('vllm:num_preemptions_total'),
    prefixQueries: sum('vllm:prefix_cache_queries_total'), prefixHits: sum('vllm:prefix_cache_hits_total'),
    totals: Object.fromEntries([...values.keys()]
      .filter(name => /(?:generation_tokens_total|prompt_tokens_total|spec_decode.*_total|prefix_cache_(?:queries|hits)_total|num_preemptions_total)$/.test(name))
      .map(name => [name, sum(name)])) };
}
