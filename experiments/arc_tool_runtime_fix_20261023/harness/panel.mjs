import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { localModel, solveTask } from './agent.mjs';
import { makeProcessBridge } from './process-bridge.mjs';
import { KvScheduler, capacityFromLog, parseMetrics } from './kv-scheduler.mjs';
import { TimeGovernor } from './time-governor.mjs';
import { taskCounts } from './failure-state.mjs';
import { CandidatePool, submissionEntries, validGrid } from './candidate-pool.mjs';

const options = {
  help: { type: 'boolean' },
  data: { type: 'string' }, output: { type: 'string' }, bridge: { type: 'string' },
  python: { type: 'string', default: 'python3' }, base: { type: 'string', default: 'http://127.0.0.1:8000/v1' },
  deadline: { type: 'string' }, concurrency: { type: 'string', default: '24' },
  'output-limit': { type: 'string', default: '65000' },
  'python-limit': { type: 'string', default: '24' },
  'python-notice-after': { type: 'string', default: '16' },
  'late-budget-notice-tokens': { type: 'string', default: '0' },
  'emergency-finalize-tokens': { type: 'string', default: '4096' },
  'emergency-repair-tokens': { type: 'string', default: '2048' },
  'initial-tasks': { type: 'string', default: '20' }, 'ramp-interval-s': { type: 'string', default: '20' },
  'cold-interval-s': { type: 'string', default: '8' },
  'pause-at': { type: 'string', default: '0.84' }, 'resume-at': { type: 'string', default: '0.78' },
  governor: { type: 'string', default: 'on' }, 'governor-reserve-s': { type: 'string', default: '900' },
  'governor-reserve-seconds': { type: 'string' },
  'min-output-limit': { type: 'string', default: '24000' }, 'governor-default-tps': { type: 'string', default: '300' },
  'reasoning-effort': { type: 'string', default: 'medium' },
  'panel-kind': { type: 'string', default: 'public48' },
  'server-log': { type: 'string' }, 'kv-capacity-tokens': { type: 'string' }, submission: { type: 'string' },
  waves: { type: 'string', default: '1' }, 'wave-seeds': { type: 'string', default: '42,1337' },
  'wave3-transform': { type: 'string', default: 'transpose' },
  'wave2-turn1-think-cap': { type: 'string', default: '0' },
  'wave2-turn-think-cap': { type: 'string', default: '0' },
  'wave2-code-reserve': { type: 'string', default: '4096' },
  'wave2-finalize-reserve': { type: 'string', default: '0' },
  'wave2-duplicate-extend': { type: 'string', default: '0' },
};
export function parsePanelArgs(argv = process.argv.slice(2)) {
  const args = parseArgs({ args: argv, options }).values;
  // Preserve the original one/two-wave defaults; only a new three-wave run
  // receives the deterministic third seed when no seed list was supplied.
  if (args.waves === '3' && !argv.some(x => x === '--wave-seeds' || x.startsWith('--wave-seeds=')))
    args['wave-seeds'] = '42,1337,2026';
  return args;
}
function write(name, value) {
  fs.writeFileSync(name + '.tmp', JSON.stringify(value));
  fs.renameSync(name + '.tmp', name);
}
// Ignore only an interrupted final JSONL write, never corrupt complete records.
function readJsonl(name) {
  if (!fs.existsSync(name)) return [];
  const lines = fs.readFileSync(name, 'utf8').split('\n'), rows = [];
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    try { rows.push(JSON.parse(lines[i])); }
    catch (error) { if (i !== lines.length - 1) throw error; }
  }
  return rows;
}

export function transposeGrid(grid) {
  if (!validGrid(grid)) return grid;
  return grid[0].map((_, column) => grid.map(row => row[column]));
}
export function transposeTask(task) {
  return { ...structuredClone(task),
    train: task.train.map(pair => ({ ...pair, input: transposeGrid(pair.input), output: transposeGrid(pair.output) })),
    // Test outputs, if present in a local source, never enter the solver.
    test: task.test.map(pair => ({ input: transposeGrid(pair.input) })) };
}
export function wrapTransposedProgram(code) {
  if (typeof code !== 'string' || !code.trim()) return code;
  // Keep the source at module scope: relocating it changes explicit global
  // declarations. During a call, recursive/helper references to transform must
  // resolve to the original solver, then the wrapper is restored even on error.
  let prefix = '_arc_wave3';
  while (code.includes(prefix)) prefix += '_x';
  return code.trim() + '\n\n'
    + `import numpy as ${prefix}_numpy\n${prefix}_solver = transform\n\n`
    + `def ${prefix}_wrapper(grid):\n`
    + `    global transform, ${prefix}_solver\n`
    + `    ${prefix}_outer = transform\n`
    + `    transform = ${prefix}_solver\n`
    + '    try:\n'
    + `        ${prefix}_result = ${prefix}_solver([list(row) for row in zip(*grid)])\n`
    + `        if isinstance(${prefix}_result, ${prefix}_numpy.ndarray):\n`
    + `            ${prefix}_result = ${prefix}_result.tolist()\n`
    + `        return [list(row) for row in zip(*${prefix}_result)]\n`
    + '    finally:\n'
    + `        ${prefix}_solver = transform\n`
    + `        transform = ${prefix}_outer\n\n`
    // Retain exactly one top-level FunctionDef named transform for the checker.
    + `transform = ${prefix}_wrapper\n`;
}
export function canonicalizeWaveCandidate(candidate, transform = 'identity') {
  if (!candidate || typeof candidate !== 'object') return candidate;
  const copy = structuredClone(candidate);
  const needsTranspose = transform === 'transpose' && copy.coordinate_frame !== 'original';
  if (needsTranspose) {
    for (const key of ['predictions', 'fallback_predictions', 'verified_fallback_predictions'])
      if (Array.isArray(copy[key])) copy[key] = copy[key].map(transposeGrid);
    if (Array.isArray(copy.attempts)) copy.attempts = copy.attempts.map(attempt => ({ ...attempt,
      attempt_1: attempt.coordinate_frame === 'original' ? attempt.attempt_1 : transposeGrid(attempt.attempt_1),
      attempt_2: attempt.coordinate_frame === 'original' ? attempt.attempt_2 : transposeGrid(attempt.attempt_2),
      coordinate_frame: 'original', wave_transform: transform }));
    for (const key of ['code', 'extracted_code', 'submitted_code', 'program'])
      if (typeof copy[key] === 'string' && copy[key].trim()) copy[key] = wrapTransposedProgram(copy[key]);
  }
  for (const key of ['candidate_pool', 'pool_candidates'])
    if (Array.isArray(copy[key])) copy[key] = copy[key].map(c => canonicalizeWaveCandidate({
      ...c, coordinate_frame: c.coordinate_frame ?? candidate.coordinate_frame }, transform));
  if (copy.verified_snapshot) copy.verified_snapshot = canonicalizeWaveCandidate({ ...copy.verified_snapshot,
    coordinate_frame: copy.verified_snapshot.coordinate_frame ?? candidate.coordinate_frame }, transform);
  return { ...copy, coordinate_frame: 'original', wave_transform: transform };
}

export class TaskRecord {
  constructor(record, manifestIndex) {
    this.record = record; this.manifest_index = manifestIndex;
    this.pool = new CandidatePool({ queryCount: record.query_count }); this.waves = {};
    this.best_demo_pass_count = 0; this.turn1_output_tokens = 0;
  }
  ingestWave(result, wave, dir, { transform = 'identity' } = {}) {
    const add = candidate => this.pool.add(canonicalizeWaveCandidate(candidate, transform), { wave }), demoCount = result.demo_count ?? 0;
    const frame = value => ({ coordinate_frame: value?.coordinate_frame, wave_transform: value?.wave_transform });
    for (const candidate of [...(result.candidate_pool ?? []), ...(result.pool_candidates ?? [])])
      add({ ...candidate, coordinate_frame: candidate.coordinate_frame ?? result.coordinate_frame });
    for (const candidate of readJsonl(path.join(dir, 'candidates.jsonl'))) add(candidate);
    for (const tool of readJsonl(path.join(dir, 'tools.jsonl'))) {
      const execution = tool.execution ?? {};
      // A Python hypothesis can pass demos yet have no legal query grids, so
      // it contributes scheduling evidence even when no pool entry was saved.
      for (const report of execution.verification_reports ?? []) {
        this.best_demo_pass_count = Math.max(this.best_demo_pass_count,
          (report.demos ?? []).filter(demo => demo.kind === 'pass').length);
      }
      for (const [candidates, parentFrame] of [[tool.pool_candidates, tool.coordinate_frame],
        [execution.pool_candidates, execution.coordinate_frame ?? tool.coordinate_frame]])
        for (const candidate of candidates ?? [])
          add({ ...candidate, coordinate_frame: candidate.coordinate_frame ?? parentFrame });
      const grids = tool.eligible ? tool.predictions : tool.candidate_predictions;
      if (grids?.length) add({ ...frame(tool), source: 'submit_program', predictions: grids, verified: !!tool.eligible,
        ...(tool.extracted_code || tool.name === 'submit_program' ? { code: tool.extracted_code || tool.code } : {}),
        action_id: tool.id, demo_count: tool.feedback?.demos?.length ?? demoCount,
        demo_pass_count: tool.demo_pass_count ?? 0, demo_diff_cells: tool.demo_diff_cells });
      const snapshot = execution.verified_snapshot;
      for (const [saved, savedFrame] of [[execution.verified_fallback_predictions, execution],
        [snapshot?.predictions ?? snapshot?.verified_fallback_predictions, snapshot]])
      if (saved?.length) add({ ...frame(savedFrame),
        coordinate_frame: savedFrame?.coordinate_frame ?? execution.coordinate_frame ?? tool.coordinate_frame,
        source: 'verify_demos', predictions: saved, verified: true,
        action_id: execution.verified_fallback_action_id ?? tool.id, demo_count: demoCount,
        demo_pass_count: demoCount, demo_diff_cells: 0 });
      this.best_demo_pass_count = Math.max(this.best_demo_pass_count, tool.demo_pass_count ?? 0);
    }
    const snapshotPath = path.join(dir, 'verified_fallback.json');
    if (fs.existsSync(snapshotPath)) {
      const snapshot = JSON.parse(fs.readFileSync(snapshotPath, 'utf8'));
      add({ ...frame(snapshot), source: 'verify_demos', verified: true, predictions: snapshot.verified_fallback_predictions ?? snapshot.predictions,
        action_id: snapshot.action_id, demo_count: demoCount, demo_pass_count: demoCount, demo_diff_cells: 0 });
    }
    const formal = result.status === 'eligible' || result.answer_source === 'submit_program'
      || result.answer_source === 'overflow_verified_program';
    if (formal || result.answer_source === 'verified_fallback' || result.verified_fallback_used)
      add({ ...frame(result), source: formal ? 'submit_program' : 'verify_demos', predictions: result.predictions,
        ...(result.code ? { code: result.code } : {}), ...(result.extracted_code ? { extracted_code: result.extracted_code } : {}),
        verified: true, demo_count: demoCount, demo_pass_count: demoCount, demo_diff_cells: 0 });
    if (result.fallback_predictions?.length) add({ ...frame(result), source: 'failed_submission', verified: false,
      predictions: result.fallback_predictions, demo_count: demoCount,
      demo_pass_count: result.fallback_demo_pass_count, demo_diff_cells: result.fallback_demo_diff_cells });
    this.best_demo_pass_count = Math.max(this.best_demo_pass_count, result.best_demo_pass_count ?? 0, result.fallback_demo_pass_count ?? 0,
      ...this.pool.candidates.map(c => c.demo_pass_count ?? 0));
    if (wave === 1) this.turn1_output_tokens = result.turn1_output_tokens ?? 0;
    this.waves[wave] = canonicalizeWaveCandidate(result, transform);
    return this.result();
  }
  hasVerified() { return this.pool.candidates.some(c => c.verified && c.tier >= 1); }
  result() {
    const entries = Object.values(this.waves), last = entries.at(-1) ?? {};
    const eligible = entries.find(x => x.status === 'eligible');
    const censored = entries.find(x => x.status === 'censored');
    const freshVerified = this.pool.candidates.some(c => c.tier === 3);
    const fallback = entries.find(x => x.answer_source === 'verified_fallback' || x.verified_fallback_used);
    const selected = this.pool.selectAttempts(this.record.query_count), verified = this.hasVerified();
    const result = { ...this.record, ...last, manifest_index: this.manifest_index,
      status: eligible || freshVerified ? 'eligible' : censored ? 'censored' : (last.status ?? 'not_dispatched'),
      eligible: !!eligible || freshVerified,
      answer_source: eligible?.answer_source ?? (freshVerified ? 'submit_program'
        : verified ? fallback?.answer_source ?? 'verified_fallback' : last.answer_source),
      verified_fallback_used: !eligible && !freshVerified && verified,
      predictions: selected.predictions, attempts: selected.attempts, attempt_sources: selected.sources,
      pool_candidates: this.pool.candidates, candidate_pool: this.pool.candidates,
      best_demo_pass_count: this.best_demo_pass_count, turn1_output_tokens: this.turn1_output_tokens,
      distinct_attempt_2: selected.attempts.filter(a => JSON.stringify(a.attempt_1) !== JSON.stringify(a.attempt_2)).length,
      wave1: this.waves[1], wave2: this.waves[2], wave3: this.waves[3], waves_completed: entries.length };
    for (const key of ['output_tokens', 'input_tokens', 'cached_input_tokens', 'reasoning_tokens',
      'requests', 'python_calls', 'final_checks', 'tool_errors', 'tool_seconds', 'seconds', 'service_seconds'])
      result[key] = entries.reduce((n, x) => n + (x[key] ?? 0), 0);
    return result;
  }
}
export function buildWave2Schedule(records) {
  return records.filter(r => !r.hasVerified()).sort((x, y) =>
    y.best_demo_pass_count - x.best_demo_pass_count || y.turn1_output_tokens - x.turn1_output_tokens
    || x.manifest_index - y.manifest_index);
}
export function hasTwoDistinctVerifiedGrids(record) {
  const result = record.result();
  return result.attempts.length === record.record.query_count && result.attempts.every((attempt, q) => {
    const { attempt_1: s1, attempt_2: s2 } = result.attempt_sources[q];
    return s1?.verified && s2?.verified && (s1.wave !== s2.wave || (s2.tier ?? 0) >= 2)
      && JSON.stringify(attempt.attempt_1) !== JSON.stringify(attempt.attempt_2);
  });
}
export function hasMultiWaveConsensus(record) {
  const result = record.result();
  const candidates = result.pool_candidates ?? result.candidate_pool ?? [];
  return result.attempts.length === record.record.query_count && result.attempts.every((attempt, q) => {
    const key = JSON.stringify(attempt.attempt_1);
    const waves = new Set(candidates
      .filter(c => c.verified && (c.tier ?? 0) >= 1 && c.wave != null && JSON.stringify(c.predictions?.[q]) === key)
      .map(c => c.wave));
    return waves.size >= 2;
  });
}
export function buildWave3Schedule(records) {
  const remaining = records.filter(r => !hasTwoDistinctVerifiedGrids(r));
  const redundant = r => r.result().distinct_attempt_2 < r.record.query_count;
  const singleWaveVerified = remaining.filter(r => r.hasVerified() && redundant(r) && !hasMultiWaveConsensus(r))
    .sort((a, b) => a.manifest_index - b.manifest_index);
  const unverified = remaining.filter(r => !r.hasVerified()).sort((a, b) =>
    b.best_demo_pass_count - a.best_demo_pass_count || b.turn1_output_tokens - a.turn1_output_tokens
    || a.manifest_index - b.manifest_index);
  const consensusVerified = remaining.filter(r => r.hasVerified() && redundant(r) && hasMultiWaveConsensus(r))
    .sort((a, b) => a.manifest_index - b.manifest_index);
  const preserved = remaining.filter(r => r.hasVerified() && !redundant(r))
    .sort((a, b) => a.manifest_index - b.manifest_index);
  return [...unverified, ...singleWaveVerified, ...consensusVerified, ...preserved];
}

// Injectable dependencies let local synthetic tests exercise actual persistence
// and scheduling without a model or GPU.
export async function runPanel(args, deps = {}) {
  for (const key of ['data', 'output', 'bridge', 'deadline']) if (!args[key]) throw new Error('--' + key + ' required');
  if (!['xhigh', 'medium', 'low'].includes(args['reasoning-effort'])) throw new Error('Invalid reasoning effort');
  if (Number(args.concurrency) !== 24) throw new Error('This panel keeps a ceiling of 24; KV admission controls the resident count');
  const outputLimit = Number(args['output-limit']), pythonLimit = Number(args['python-limit']);
  if (!Number.isInteger(pythonLimit) || pythonLimit < 1 || pythonLimit > 30) throw new Error('Expected --python-limit in 1..30');
  if (!Number.isInteger(outputLimit) || outputLimit <= 0) throw new Error('Expected positive --output-limit');
  const pythonNoticeAfter = Number(args['python-notice-after']);
  if (!Number.isInteger(pythonNoticeAfter) || pythonNoticeAfter < 0) throw new Error('Expected --python-notice-after >= 0');
  if (!['on', 'off'].includes(args.governor)) throw new Error('Expected --governor on|off');
  const waves = Number(args.waves), waveSeeds = args['wave-seeds'].split(',').map(x => Number(x.trim()));
  if (![1, 2, 3].includes(waves)) throw new Error('Expected --waves 1|2|3');
  if (waveSeeds.length < waves || waveSeeds.some(x => !Number.isSafeInteger(x))) throw new Error('Expected integer --wave-seeds for each wave');
  const wave3Transform = args['wave3-transform'] ?? 'transpose';
  if (!['identity', 'transpose'].includes(wave3Transform)) throw new Error('Expected --wave3-transform identity|transpose');
  const emergencyControls = Object.fromEntries([
    ['lateBudgetNoticeTokens', 'late-budget-notice-tokens', 0],
    ['emergencyFinalizeTokens', 'emergency-finalize-tokens', 4096],
    ['emergencyRepairTokens', 'emergency-repair-tokens', 2048],
  ].map(([key, flag, fallback]) => {
    const value = Number(args[flag] ?? fallback);
    if (!Number.isInteger(value) || value < 0) throw new Error('Expected --' + flag + ' >= 0');
    return [key, value];
  }));
  const wave2Controls = Object.fromEntries([
    ['turn1ThinkCap', 'wave2-turn1-think-cap'], ['turnThinkCap', 'wave2-turn-think-cap'],
    ['codeReserve', 'wave2-code-reserve'], ['finalizeReserve', 'wave2-finalize-reserve'],
    ['duplicateExtendTokens', 'wave2-duplicate-extend'],
  ].map(([key, flag]) => {
    const value = Number(args[flag]);
    if (!Number.isInteger(value) || value < 0) throw new Error('Expected --' + flag + ' >= 0');
    return [key, value];
  }));
  const deadlineMs = Number(args.deadline) * 1000;
  if (!Number.isFinite(deadlineMs) || deadlineMs <= 0) throw new Error('Expected positive --deadline');
  const output = path.resolve(args.output); fs.mkdirSync(output, { recursive: true });
  const manifest = JSON.parse(fs.readFileSync(path.join(args.data, 'manifest.json'), 'utf8'));
  const taskCount = manifest.tasks?.length ?? manifest.count;
  const queryCount = manifest.query_count ?? manifest.tasks.reduce((n, x) => n + x.query_count, 0);
  if (!['public48', 'kaggle'].includes(args['panel-kind'])) throw new Error('Expected --panel-kind public48|kaggle');
  if (args['panel-kind'] === 'public48' && (taskCount !== 48 || queryCount !== 67)) throw new Error('Expected fixed public 48-task / 67-query panel');
  if (!(taskCount > 0) || manifest.tasks.length !== taskCount) throw new Error('Expected non-empty ARC panel');
  if (new Set(manifest.tasks.map(r => r.task_id)).size !== taskCount) throw new Error('Expected unique task IDs');
  const controller = deps.controller ?? new AbortController();
  const capacityTokens = deps.scheduler ? undefined : args['kv-capacity-tokens'] === undefined
    ? capacityFromLog(fs.readFileSync(args['server-log'] ?? path.join(path.dirname(output), 'server.log'), 'utf8'))
    : Number(args['kv-capacity-tokens']);
  const scheduler = deps.scheduler ?? new KvScheduler({ capacityTokens, maxTasks: 24, signal: controller.signal,
    initialTasks: Number(args['initial-tasks']), rampIntervalMs: Number(args['ramp-interval-s']) * 1000,
    minColdIntervalMs: Number(args['cold-interval-s']) * 1000,
    pauseAt: Number(args['pause-at']), resumeAt: Number(args['resume-at']) });
  const governor = deps.governor ?? (args.governor === 'on' ? new TimeGovernor({ deadlineMs,
    totalTasks: taskCount, outputLimit, minLimit: Math.min(outputLimit, Number(args['min-output-limit'])),
    reserveMs: Number(args['governor-reserve-seconds'] ?? args['governor-reserve-s']) * 1000,
    initialTokensPerSec: Number(args['governor-default-tps']) }) : null);
  const timer = setTimeout(() => controller.abort('save_cutoff'), Math.max(0, deadlineMs - Date.now()));
  const onTerm = () => controller.abort('SIGTERM'), onInt = () => controller.abort('SIGINT');
  process.on('SIGTERM', onTerm); process.on('SIGINT', onInt);
  const started = Date.now(), results = {}, active = new Map();
  const records = manifest.tasks.map((record, i) => new TaskRecord(record, i));
  let cursor = 0, currentWave = 1, firstGeneration = false, engineFailure = null;
  let runFinished = false, wavesFinished = 0, wave2Schedule = [], wave2Skipped = [];
  const wave2Claimed = new Set(), workWaiters = new Set(), WAIT_FOR_WORK = Symbol('wait_for_work');
  let wave2AdmissionBlocked = false, wave2ReadyDirty = true, wave2Ready = [];
  const wave3Claimed = new Set();
  let wave3AdmissionBlocked = false, wave3ReadyDirty = true, wave3Ready = [], wave3Schedule = [], wave3Skipped = [];
  let wave3BaselineCompleted = false;
  const isBaselineSettled = r => Boolean(r.waves[1] && (r.waves[2] || r.hasVerified()));
  function hasWave1ColdStartPending() {
    if (cursor < records.length) return true;
    for (const [id, info] of active.entries()) {
      if (info.wave !== 1) continue;
      if ((info.requests ?? 0) === 0 && (info.phase === 'waiting_for_kv' || info.phase === 'preparing')) return true;
      if (scheduler.queue?.some?.(q => q.taskId === id && !q.continuation)) return true;
    }
    return false;
  }
  function wakeWorkers() {
    wave2AdmissionBlocked = false;
    wave3AdmissionBlocked = false;
    for (const resolve of workWaiters) resolve();
    workWaiters.clear();
  }
  controller.signal.addEventListener('abort', wakeWorkers);
  function failEngine(error) {
    if (engineFailure) return;
    engineFailure = { at: Date.now() / 1000, error: String(error), automatic_restart: false };
    write(path.join(output, 'ENGINE_FAILURE.json'), engineFailure); controller.abort('engine_failure');
  }
  const onEngineExit = () => failEngine('model_server_process_exited');
  process.on('SIGUSR2', onEngineExit);
  function summary() {
    // Wave 2 has an existing Wave 1 record; count each task ID once.
    const counts = taskCounts(results, Object.fromEntries([...active].filter(([id]) => !results[id])), taskCount);
    const completedTokens = Object.values(results).reduce((n, x) => n + (x.output_tokens ?? 0), 0);
    const inFlightTokens = [...active.values()].reduce((n, x) => n + (x.output_tokens ?? 0) + (x.current_usage?.output ?? 0), 0);
    const panelStatus = engineFailure ? 'failed_partial' : !runFinished && !controller.signal.aborted ? 'running'
      // Optional waves cannot undo a fully completed panel once cleanup drains.
      : runFinished && wavesFinished >= 1 && counts.completed_tasks === taskCount ? 'completed'
      : controller.signal.aborted ? 'partial' : counts.completed_tasks === taskCount ? 'completed' : 'failed_partial';
    if (args.submission) write(args.submission, Object.fromEntries(manifest.tasks.map(record =>
      [record.task_id, submissionEntries(results[record.task_id]?.attempts, record.query_count)])));
    const value = { status: panelStatus, started_at: started / 1000, seconds: (Date.now() - started) / 1000,
      reasoning_effort: args['reasoning-effort'], output_limit: outputLimit, python_limit: pythonLimit,
      waves, wave_seeds: waveSeeds.slice(0, waves), current_wave: currentWave, waves_finished: wavesFinished,
      wave2_controls: wave2Controls, wave2_schedule: wave2Schedule, wave2_skipped: wave2Skipped,
      emergency_controls: emergencyControls, wave3_transform: wave3Transform,
      wave3_schedule: wave3Schedule, wave3_skipped: wave3Skipped,
      ...counts, engine_failure: engineFailure, active_tasks: [...scheduler.leases.keys()],
      queued_tasks: scheduler.queue.map(x => x.taskId), logical_inflight_tasks: [...active.keys()],
      not_scheduled_tasks: taskCount - cursor,
      wave_not_scheduled_tasks: currentWave === 1 ? taskCount - cursor : currentWave === 2
        ? taskCount - wave2Claimed.size - wave2Skipped.length : taskCount - wave3Claimed.size - wave3Skipped.length,
      output_tokens: completedTokens,
      in_flight_output_tokens: inFlightTokens, total_live_output_tokens: completedTokens + inFlightTokens,
      scheduling: scheduler.snapshot(), governor: governor?.snapshot() ?? null,
      running: Object.fromEntries(active), tasks: results };
    write(path.join(output, 'summary.json'), value); return value;
  }
  function makeBridge(taskFile, dir, task) {
    if (deps.makeBridge) return deps.makeBridge(taskFile, dir, task);
    return makeProcessBridge({ command: args.python, args: ['-u', args.bridge, '--task', taskFile, '--output', dir],
      stderr: fs.createWriteStream(path.join(dir, 'bridge.stderr.log')), signal: controller.signal });
  }
  function claimWave3() {
    if (waves < 3) return active.size ? WAIT_FOR_WORK : null;
    if (records.some(r => !isBaselineSettled(r) && !active.has(r.record.task_id))) {
      if (!wave3Skipped.length && !wave3Claimed.size) {
        wave3Ready = [];
        wave3Skipped = records.map(r => ({ task_id: r.record.task_id, reason: 'baseline_incomplete', at: Date.now() / 1000 }));
        summary();
      }
      return active.size ? WAIT_FOR_WORK : null;
    }
    if (wave3ReadyDirty) {
      for (const r of records) {
        if (isBaselineSettled(r) && !active.has(r.record.task_id) && !wave3Claimed.has(r.record.task_id)
          && !wave3Skipped.some(s => s.task_id === r.record.task_id) && hasTwoDistinctVerifiedGrids(r)) {
          wave3Skipped.push({ task_id: r.record.task_id, reason: 'two_distinct_verified_grids', at: Date.now() / 1000 });
        }
      }
      wave3Ready = buildWave3Schedule(records.filter(r => isBaselineSettled(r) && !active.has(r.record.task_id)
        && !wave3Claimed.has(r.record.task_id) && !wave3Skipped.some(s => s.task_id === r.record.task_id)));
      wave3ReadyDirty = false;
      const allBaselineSettled = records.every(isBaselineSettled);
      if (allBaselineSettled && !wave3BaselineCompleted) {
        wave3BaselineCompleted = true;
        // The barrier receipt is committed once every baseline (Wave 1/2) task has settled.
        write(path.join(output, 'WAVE3_BASELINE_COMPLETE.json'), { at: Date.now() / 1000,
          task_count: records.length, waves_finished: 2,
          schedule: [...wave3Schedule, ...wave3Ready.map(r => r.record.task_id)] });
      }
    }
    currentWave = 3;
    if (!wave3Ready.length || wave3AdmissionBlocked) return active.size ? WAIT_FOR_WORK : null;
    const taskRecord = wave3Ready[0];
    if (taskRecord.hasVerified() && records.some(r => !isBaselineSettled(r))) return WAIT_FOR_WORK;
    const decision = governor?.allocate(taskRecord.record.task_id, { wave: 3 });
    if (decision?.admit === false) {
      if (active.size) { wave3AdmissionBlocked = true; return WAIT_FOR_WORK; }
      wave3Skipped.push(...wave3Ready.map(r => ({ task_id: r.record.task_id, reason: decision.reason, at: Date.now() / 1000 })));
      wave3Ready = []; summary(); return null;
    }
    wave3Ready.shift(); wave3Claimed.add(taskRecord.record.task_id); wave3Schedule.push(taskRecord.record.task_id);
    return { taskRecord, wave: 3, decision };
  }
  function claimWork() {
    // Synchronous persistence can delay timer delivery past the cutoff.
    if (Date.now() >= deadlineMs && !controller.signal.aborted) controller.abort('save_cutoff');
    if (controller.signal.aborted) return null;
    // Claim every Wave 1 task before spending a worker slot on optional work.
    if (cursor < records.length) return { taskRecord: records[cursor++], wave: 1 };
    if (waves === 1) return null;
    if (hasWave1ColdStartPending()) return WAIT_FOR_WORK;
    if (currentWave === 3) return claimWave3();
    currentWave = 2;
    // A record becomes ready only after its first result and merged candidates
    // have been saved and its bridge/lease have finished cleanup.
    if (wave2ReadyDirty) {
      wave2Ready = buildWave2Schedule(records.filter(r => r.waves[1] && !wave2Claimed.has(r.record.task_id)
        && !wave2Skipped.some(s => s.task_id === r.record.task_id)));
      wave2ReadyDirty = false;
    }
    if (wave2Ready.length && !wave2AdmissionBlocked) {
      const taskRecord = wave2Ready[0];
      const decision = governor?.allocate(taskRecord.record.task_id, { wave: 2 });
      if (decision?.admit === false) {
        // A straggler may release unused Wave 1 allowance when it finishes.
        // Do not turn a temporary refusal into a permanent skip of all Wave 2.
        if (active.size) { wave2AdmissionBlocked = true; return WAIT_FOR_WORK; }
        wave2Skipped.push(...wave2Ready.map(r => ({ task_id: r.record.task_id,
          reason: decision.reason, at: Date.now() / 1000 })));
        wave2Ready = [];
        summary(); return claimWave3();
      }
      wave2Ready.shift();
      wave2Claimed.add(taskRecord.record.task_id);
      // This is the actual claim order among ready tasks, not a plan containing
      // unfinished Wave 1 tasks whose priority evidence is still unknown.
      wave2Schedule.push(taskRecord.record.task_id);
      return { taskRecord, wave: 2, decision };
    }
    if (wave2Ready.length || records.some(r => !r.waves[1])) return active.size ? WAIT_FOR_WORK : claimWave3();
    return claimWave3();
  }
  async function worker() {
    while (!controller.signal.aborted) {
      // Claiming and registering active work are synchronous, so no worker can
      // start the same task twice or overlap its first and second passes.
      const work = claimWork();
      if (work === WAIT_FOR_WORK) { await new Promise(resolve => workWaiters.add(resolve)); continue; }
      if (!work) break;
      const { taskRecord, wave } = work, record = taskRecord.record;
      const decision = work.decision ?? governor?.allocate(record.task_id, { wave });
      const cap = Math.min(outputLimit, decision?.limit ?? outputLimit);
      governor?.startTask(record.task_id, cap, { wave });
      const taskDir = path.join(output, record.task_id), dir = wave === 1 ? taskDir : path.join(taskDir, 'wave' + wave);
      let lastPartial = 0, turn1Tokens = 0, result, task, events, bridge;
      active.set(record.task_id, { wave, requests: 0, output_tokens: 0, phase: deps.solveTask ? 'running' : 'waiting_for_kv' }); summary();
      try {
        fs.mkdirSync(dir, { recursive: true });
        let taskFile = path.resolve(args.data, record.path);
        task = JSON.parse(fs.readFileSync(taskFile, 'utf8'));
        if (wave === 3 && wave3Transform === 'transpose') {
          task = transposeTask(task); taskFile = path.join(dir, 'task_transposed.json'); write(taskFile, task);
        }
        events = fs.createWriteStream(path.join(dir, 'events.jsonl'));
        bridge = makeBridge(taskFile, dir, task);
        result = await (deps.solveTask ?? solveTask)({ task, taskFile, taskId: record.task_id, scheduler, wave, seed: waveSeeds[wave - 1],
          outputLimit: cap, pythonLimit, pythonNoticeAfter, ...emergencyControls, governor, governorDecision: decision,
          reasoningEffort: args['reasoning-effort'], onBackendFailure: failEngine,
          ...(wave === 2 ? { ...wave2Controls, previousPredictions: results[record.task_id]?.attempts ?? null } : {}),
          model: localModel(args.base), signal: controller.signal, bridge, emit: async (event, state) => {
            if (state) {
              const prev = active.get(record.task_id);
              const sameSettled = prev && prev.output_tokens === state.output_tokens && prev.requests === state.requests && event.type !== 'message_end';
              active.set(record.task_id, { wave, requests: state.requests, output_tokens: state.output_tokens,
                input_tokens: state.input_tokens, cached_input_tokens: state.cached_input_tokens,
                cache: state.cache, phase: state.phase, admission_wait_seconds: state.admission_wait_seconds,
                ...(sameSettled && prev.current_usage ? { current_usage: prev.current_usage } : {}) });
            }
            if (event.type === 'request') {
              const { messages, ...controls } = event.payload;
              write(path.join(dir, 'request_' + event.request_number + '.json'), { at: event.at,
                request_number: event.request_number, controls, cache_plan: event.cache_plan, message_count: messages.length });
            } else if (event.type === 'message_update') {
              if (!firstGeneration && event.assistantMessageEvent?.delta) {
                firstGeneration = true;
                write(path.join(output, 'FIRST_GENERATION_OBSERVED.json'), { at: Date.now() / 1000, task_id: record.task_id, wave });
              }
              // Live accounting is independent of the periodic disk snapshots.
              if (event.message?.usage) active.set(record.task_id, { ...active.get(record.task_id), current_usage: event.message.usage });
              if (Date.now() - lastPartial > 30000) {
                write(path.join(dir, 'active_response.json'), event.message); lastPartial = Date.now();
              }
            } else if (event.type !== 'agent_end') {
              events.write(JSON.stringify(event) + '\n');
              if (event.type === 'kv_admitted' && wave === 1) wakeWorkers();
              if (event.type === 'message_end' && event.message?.role === 'assistant' && state?.requests === 1)
                turn1Tokens = event.message.usage?.output ?? 0;
            }
          } });
      } catch (error) {
        const partial = active.get(record.task_id) ?? {};
        result = { ...partial, output_tokens: (partial.output_tokens ?? 0) + (partial.current_usage?.output ?? 0),
          status: controller.signal.aborted ? 'deadline' : 'task_error', stop_reason: String(error),
          pool_candidates: bridge?.poolCandidates ?? [] };
      } finally {
        try { await bridge?.close(); }
        catch (error) { result = { ...result, bridge_close_error: String(error) }; }
        if (events) await new Promise(resolve => events.end(resolve));
      }
      result = { ...result, wave, seed: waveSeeds[wave - 1], demo_count: task?.train?.length ?? 0,
        coordinate_frame: wave === 3 && wave3Transform === 'transpose' ? 'transposed' : 'original',
        wave_transform: wave === 3 ? wave3Transform : 'identity',
        turn1_output_tokens: result.turn1_output_tokens ?? turn1Tokens };
      governor?.finishTask(record.task_id, result.output_tokens, { wave });
      write(path.join(taskDir, 'wave' + wave + '_result.json'), result);
      if (wave > 1) write(path.join(dir, 'result.json'), result);
      results[record.task_id] = taskRecord.ingestWave(result, wave, dir, { transform: wave === 3 ? wave3Transform : 'identity' });
      active.delete(record.task_id);
      if (wave === 1) {
        if (waves >= 2 && taskRecord.hasVerified() && !wave2Skipped.some(s => s.task_id === record.task_id))
          wave2Skipped.push({ task_id: record.task_id, reason: 'wave1_verified', at: Date.now() / 1000 });
        wave2ReadyDirty = true;
      }
      if (wave <= 2) wave3ReadyDirty = true;
      if (records.every(r => r.waves[1]) && !controller.signal.aborted) wavesFinished = Math.max(wavesFinished, 1);
      if (waves >= 2 && records.every(isBaselineSettled) && !controller.signal.aborted) wavesFinished = Math.max(wavesFinished, 2);
      write(path.join(taskDir, 'result.json'), results[record.task_id]); summary();
      wakeWorkers();
    }
  }
  let readingMetrics = false;
  async function observeKv() {
    if (readingMetrics || controller.signal.aborted || deps.observeMetrics === false) return;
    readingMetrics = true;
    try {
      const sampledAt = Date.now();
      const response = await fetch(args.base.replace(/\/v1\/?$/, '') + '/metrics', {
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]) });
      if (!response.ok) throw new Error('metrics_http_' + response.status);
      const metrics = parseMetrics(await response.text()); scheduler.observe(metrics, { sampledAt });
      governor?.recordThroughputSample(Date.now(), metrics.totals?.['vllm:generation_tokens_total']);
      wakeWorkers();
      write(path.join(output, 'KV_ADMISSION.json'), scheduler.snapshot());
      fs.appendFileSync(path.join(output, 'SERVER_METRICS.jsonl'), JSON.stringify({ at: Date.now() / 1000, ...metrics }) + '\n');
    } catch (error) {
      if (!controller.signal.aborted) write(path.join(output, 'KV_OBSERVATION_ERROR.json'), { at: Date.now() / 1000, error: String(error) });
    } finally { readingMetrics = false; }
  }
  const ticker = setInterval(summary, 30000), observer = setInterval(observeKv, 5000);
  try {
    await observeKv();
    summary();
    // Keep idle workers alive across the wave boundary; a completion wakes them
    // to claim ready Wave 2 work while the final Wave 1 tasks are still running.
    // A persistence failure aborts/wakes peers, then all settle before cleanup.
    const outcomes = await Promise.allSettled(Array.from({ length: 24 }, () => worker().catch(error => {
      controller.abort('panel_worker_failure'); throw error;
    })));
    const failed = outcomes.find(x => x.status === 'rejected');
    if (failed) throw failed.reason;
    if (!controller.signal.aborted && !(waves === 3 && !records.every(isBaselineSettled))) wavesFinished = waves;
  } finally {
    currentWave = Math.min(currentWave, waves); runFinished = true;
    clearInterval(ticker); clearInterval(observer); clearTimeout(timer); scheduler.close();
    process.off('SIGTERM', onTerm); process.off('SIGINT', onInt); process.off('SIGUSR2', onEngineExit);
    controller.signal.removeEventListener('abort', wakeWorkers);
  }
  return summary();
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = parsePanelArgs();
  if (args.help) console.log('node panel.mjs --data PANEL --output RESULTS/panel --bridge bridge.py --python PYTHON --deadline UNIX_SECONDS [--waves 1|2|3 --wave-seeds 42,1337,2026 --wave3-transform transpose|identity] [--late-budget-notice-tokens 0 --emergency-finalize-tokens 4096 --emergency-repair-tokens 2048] [--wave2-turn1-think-cap 16000 --wave2-turn-think-cap 22000 --wave2-finalize-reserve 4608 --wave2-duplicate-extend 8192]');
  else {
    const result = await runPanel(args);
    if (result.engine_failure || result.status !== 'completed' || Object.values(result.tasks).some(x => x.status === 'task_error')) process.exitCode = 2;
  }
}
