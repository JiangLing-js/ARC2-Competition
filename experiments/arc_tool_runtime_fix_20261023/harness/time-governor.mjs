// Wave 1 admission estimates usage; optional waves and extensions reserve full caps.
export class TimeGovernor {
  constructor({ deadlineMs, totalTasks, outputLimit, minLimit = 24_000,
    reserveMs = 15 * 60_000, initialTokensPerSec, defaultTps, minSampleSpanMs = 120_000,
    singleSeqDivisor = 20, minSingleSeqTps = 10, wave2DrainSeconds = 60, wave1UsageRatio = 0.65, now = Date.now }) {
    if (!(deadlineMs > 0)) throw new Error('TimeGovernor needs a deadline');
    if (!(minLimit > 0 && minLimit <= outputLimit)) throw new Error('minLimit must be in (0, outputLimit]');
    const initial = initialTokensPerSec ?? defaultTps ?? 300;
    if (!(Number.isFinite(initial) && initial > 0)) throw new Error('initialTokensPerSec must be positive and finite');
    wave1UsageRatio = Number(wave1UsageRatio);
    if (!(Number.isFinite(wave1UsageRatio) && wave1UsageRatio > 0 && wave1UsageRatio <= 1))
      throw new Error('wave1UsageRatio must be in (0, 1]');
    if (!(Number.isFinite(singleSeqDivisor) && singleSeqDivisor > 0)) throw new Error('singleSeqDivisor must be positive and finite');
    if (!(Number.isFinite(minSingleSeqTps) && minSingleSeqTps > 0)) throw new Error('minSingleSeqTps must be positive and finite');
    if (!(Number.isFinite(wave2DrainSeconds) && wave2DrainSeconds >= 0)) throw new Error('wave2DrainSeconds must be non-negative and finite');
    Object.assign(this, { deadlineMs, totalTasks, outputLimit, minLimit, reserveMs,
      initialTokensPerSec: initial, minSampleSpanMs, singleSeqDivisor, minSingleSeqTps, wave2DrainSeconds, wave1UsageRatio, now });
    this.lastFullLoadRate = initial;
    this.lastFullLoadSampleAt = null;
    this.lastDrainSampleAt = null;
    this.samples = []; // External cumulative generation telemetry.
    this.localSamples = [[now(), 0]];
    this.totalGeneratedTokens = 0;
    this.active = new Map(); this.finished = [];
    this.started = 0; // Wave 1 starts only: Wave 2 never reduces its reserve.
    this.startedWave1 = new Set(); this.decisions = [];
  }
  appendSample(samples, at, total) {
    if (!Number.isFinite(at) || !Number.isFinite(total) || total < 0) return;
    const previous = samples.at(-1);
    if (previous && at < previous[0]) return;
    if (previous && total < previous[1]) samples.length = 0; // Counter reset.
    if (samples.at(-1)?.[0] === at) samples[samples.length - 1] = [at, total];
    else samples.push([at, total]);
    // Retain one point before the window for boundary interpolation.
    while (samples.length > 2 && samples[1][0] < at - 15 * 60_000) samples.shift();
  }
  recordThroughputSample(atMs, totalGeneratedTokens) {
    this.appendSample(this.samples, atMs, totalGeneratedTokens);
    if (this.samples.at(-1)?.[0] === atMs) {
      this.latestSampleActiveCount = this.active.size;
      if (this.active.size < 8 && (this.active.size > 0 || this.finished.length > 0)) {
        this.lastDrainSampleAt = atMs;
      }
    }
    this.rememberFullLoadRate(this.samples, atMs);
  }
  observeGeneration(totalTokens, at = this.now()) { this.recordThroughputSample(at, totalTokens); }
  windowRate(samples, windowMs, at = this.now(), minStartAt = 0) {
    if (samples.length < 2) return null;
    // Missing telemetry is not evidence that the previous generation rate
    // continues forever. Age both windows to now using only observed tokens.
    // This also includes idle time between task-completion fallback samples.
    const last = samples.at(-1);
    if (at > last[0]) samples = [...samples, [at, last[1]]];
    const [endAt, endTokens] = samples.at(-1), cutoff = Math.max(samples[0][0], minStartAt, endAt - windowMs);
    if (endAt - cutoff < this.minSampleSpanMs) return null;
    let startTokens = samples[0][1];
    for (let i = 1; i < samples.length; i++) {
      const [beforeAt, beforeTokens] = samples[i - 1], [afterAt, afterTokens] = samples[i];
      if (afterAt >= cutoff) {
        startTokens = beforeTokens + (afterTokens - beforeTokens) * (cutoff - beforeAt) / (afterAt - beforeAt); break;
      }
    }
    return Math.max(0, (endTokens - startTokens) / ((endAt - cutoff) / 1000));
  }
  rememberFullLoadRate(samples, at) {
    // Admission alone must not reclassify a tail sample as full-load telemetry,
    // and the first minSampleSpanMs after a low-concurrency drain must not mix
    // earlier drain samples into the 5m full-load rate.
    if (this.active.size < 8 || samples.at(-1)?.[0] !== at || this.lastFullLoadSampleAt === at) return;
    if (this.lastDrainSampleAt !== null && at - this.lastDrainSampleAt < this.minSampleSpanMs) return;
    const tps5m = this.windowRate(samples, 5 * 60_000, at, this.lastDrainSampleAt ?? 0);
    const tps15m = this.windowRate(samples, 15 * 60_000, at);
    if (tps5m !== null && tps15m !== null) {
      this.lastFullLoadRate = Math.min(tps5m, tps15m);
      this.lastFullLoadSampleAt = at;
    }
  }
  throughput(at = this.now()) {
    const samples = this.samples.length ? this.samples : this.localSamples;
    const tps5m = this.windowRate(samples, 5 * 60_000, at), tps15m = this.windowRate(samples, 15 * 60_000, at);
    const postDrainRecovery = this.lastDrainSampleAt !== null && at - this.lastDrainSampleAt < this.minSampleSpanMs;
    const tps5mEffective = (!postDrainRecovery && this.lastDrainSampleAt !== null)
      ? (this.windowRate(samples, 5 * 60_000, at, this.lastDrainSampleAt) ?? tps5m)
      : tps5m;
    const measured = tps5mEffective === null || tps15m === null ? this.initialTokensPerSec : Math.min(tps5mEffective, tps15m);
    // Include the zero-active handoff after a wave drains and the initial post-drain
    // ramp window before minSampleSpanMs of fresh full-load telemetry has accumulated.
    const draining = ((this.active.size < 8 || this.latestSampleActiveCount < 8)
      && (this.active.size > 0 || this.finished.length > 0)) || postDrainRecovery;
    return { tps_5m: tps5m, tps_15m: tps15m,
      r_safe: draining ? Math.max(measured, this.lastFullLoadRate) : measured };
  }
  get tps() { return this.throughput().r_safe; }
  projection(at = this.now()) {
    const remainingSeconds = Math.max(0, (this.deadlineMs - this.reserveMs - at) / 1000), rates = this.throughput(at);
    const activeRemainingObligation = [...this.active.values()].reduce((n, x) => n + Math.max(0, x.limit - x.used), 0);
    const unstarted = Math.max(0, this.totalTasks - this.started), unstartedWave1Reserve = unstarted * this.minLimit;
    const grossSupply = rates.r_safe * remainingSeconds, S = grossSupply - activeRemainingObligation - unstartedWave1Reserve;
    const singleSeqTps = Math.max(this.minSingleSeqTps, rates.r_safe / this.singleSeqDivisor);
    const wave2SingleTaskAllowance = Math.max(0, (remainingSeconds - this.wave2DrainSeconds) * singleSeqTps);
    return { remainingSeconds, seconds: remainingSeconds, ...rates, tps: rates.r_safe,
      grossSupply, activeRemainingObligation, activeDemand: activeRemainingObligation,
      unstarted, unstartedWave1Reserve, S, supply: S, singleSeqTps, wave2SingleTaskAllowance };
  }
  computeWaveCap({ wave = 1, unstarted = 0, at = this.now() } = {}) {
    const p = this.projection(at);
    if (wave !== 1) return { ...p, allowance: Math.min(p.S, p.wave2SingleTaskAllowance) };
    const expectedUsage = this.outputLimit * this.wave1UsageRatio;
    const activeRemainingObligation = [...this.active.values()].reduce((n, task) => {
      const extensions = task.extension_tokens ?? 0;
      const expected = task.wave === 1 ? Math.min(task.limit - extensions, expectedUsage) : task.limit - extensions;
      return n + Math.max(0, expected - task.used) + extensions;
    }, 0);
    const unstartedExpectedObligation = unstarted * expectedUsage;
    const allowance = this.outputLimit;
    const S = p.grossSupply - activeRemainingObligation - p.unstartedWave1Reserve;
    return { ...p, activeRemainingObligation, activeDemand: activeRemainingObligation, S, supply: S,
      unstartedExpectedObligation, wave1UsageRatio: this.wave1UsageRatio, allowance };
  }
  allocate(taskId, { wave = 1, at = this.now() } = {}) {
    const p = this.projection(at), running = this.active.get(taskId);
    if (running) {
      if (running.wave !== wave) throw new Error(`Task ${taskId} is already active in wave ${running.wave}`);
      const runningLimit = wave === 1 ? this.outputLimit : running.limit;
      return { task_id: taskId, wave, at, admit: true, limit: runningLimit,
        reduced: runningLimit < this.outputLimit, throttled: runningLimit < this.outputLimit,
        reason: 'already_started', ...p };
    }
    // Wave 1 is hard-locked to full outputLimit (65,000). Optional waves still fit full
    // outstanding caps and the time available to one sequence before drain.
    const cap = this.computeWaveCap({ wave, unstarted: p.unstarted, at }), allowance = cap.allowance;
    const refused = wave >= 2 && (p.remainingSeconds <= this.wave2DrainSeconds || allowance < this.minLimit);
    const limit = wave === 1 ? this.outputLimit
      : (refused ? 0 : Math.max(this.minLimit, Math.min(this.outputLimit, Math.floor(allowance))));
    const decision = { task_id: taskId, wave, at, admit: !refused, limit,
      reduced: limit < this.outputLimit, throttled: limit < this.outputLimit,
      reason: wave === 1 ? 'wave1_locked_full' : refused ? `wave${wave}_budget_insufficient`
        : limit < this.outputLimit ? 'remaining_supply_scaled' : 'full_budget',
      per_task_tokens: wave === 1 ? this.outputLimit : Math.floor(allowance), ...cap };
    this.decisions.push(decision); return decision;
  }
  startTask(taskId, cap, { wave = 1 } = {}) {
    if (!(cap > 0 && cap <= this.outputLimit)) throw new Error('Task cap must be in (0, outputLimit]');
    const running = this.active.get(taskId);
    if (running) {
      if (running.wave !== wave || running.limit !== cap) throw new Error(`Task ${taskId} already started with different allocation`);
      return running;
    }
    if (wave === 1 && !this.startedWave1.has(taskId)) { this.startedWave1.add(taskId); this.started++; }
    const task = { task_id: taskId, wave, cap, limit: cap, used: 0 }; this.active.set(taskId, task); return task;
  }
  grantExtension(taskId, tokens, { wave = 1, reason = 'emergency_finalize', at = this.now() } = {}) {
    if (!Number.isInteger(tokens) || tokens < 0) throw new Error('Extension tokens must be a non-negative integer');
    const task = this.active.get(taskId), p = this.projection(at);
    if (task && task.wave !== wave) throw new Error(`Task ${taskId} extends wrong wave`);
    // Adding to the active limit immediately reserves the new obligation. It
    // cannot consume other active tasks' budgets or unstarted Wave 1 minima.
    const remaining = task ? Math.max(0, task.limit - task.used) : 0;
    const timeSpare = Math.max(0, p.wave2SingleTaskAllowance - remaining);
    const granted = task ? Math.max(0, Math.min(tokens, Math.floor(p.S), Math.floor(timeSpare))) : 0;
    if (granted) {
      task.limit += granted; task.cap = task.limit;
      task.extension_tokens = (task.extension_tokens ?? 0) + granted;
    }
    const decision = { task_id: taskId, wave, at, extension: true, requested_tokens: tokens,
      granted_tokens: granted, limit: task?.limit ?? 0,
      reason: granted ? reason : task ? 'extension_budget_insufficient' : 'task_not_active', ...p };
    this.decisions.push(decision);
    return decision;
  }
  updateLiveUsage(taskId, liveTokens) {
    const task = this.active.get(taskId);
    if (!task || !Number.isFinite(liveTokens)) return;
    const next = Math.max(task.used, liveTokens, 0);
    this.totalGeneratedTokens += next - task.used; task.used = next;
    this.appendSample(this.localSamples, this.now(), this.totalGeneratedTokens);
    if (!this.samples.length) {
      this.latestSampleActiveCount = this.active.size;
      if (this.active.size < 8 && (this.active.size > 0 || this.finished.length > 0)) {
        this.lastDrainSampleAt = this.now();
      }
      this.rememberFullLoadRate(this.localSamples, this.now());
    }
  }
  finishTask(taskId, finalTokens, { wave } = {}) {
    const task = this.active.get(taskId);
    if (!task) return;
    if (wave !== undefined && wave !== task.wave) throw new Error(`Task ${taskId} finishes wrong wave`);
    this.updateLiveUsage(taskId, finalTokens ?? task.used); this.active.delete(taskId); this.finished.push({ ...task });
  }
  // Legacy single-wave API.
  admit(taskId, at = this.now()) {
    const decision = this.allocate(taskId, { wave: 1, at }); this.startTask(taskId, decision.limit, { wave: 1 }); return decision;
  }
  progress(taskId, used) { this.updateLiveUsage(taskId, used); }
  finish(taskId, used) { this.finishTask(taskId, used); }
  snapshot() {
    return { deadline_ms: this.deadlineMs, reserve_ms: this.reserveMs, output_limit: this.outputLimit,
      min_limit: this.minLimit, wave1_usage_ratio: this.wave1UsageRatio,
      last_full_load_rate: this.lastFullLoadRate, started: this.started, finished: this.finished.length,
      total_generated_tokens: this.totalGeneratedTokens, active: Object.fromEntries(this.active),
      finished_by_wave: this.finished.reduce((counts, x) => ({ ...counts, [x.wave]: (counts[x.wave] ?? 0) + 1 }), {}),
      projection: this.projection(), reduced_admissions: this.decisions.filter(d => d.admit && d.reduced).length,
      refused_admissions: this.decisions.filter(d => !d.extension && !d.admit).length, last_decision: this.decisions.at(-1) ?? null };
  }
}
