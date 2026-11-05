// Silent pool selection preserves independent wave representatives.
export function validGrid(grid) {
  return Array.isArray(grid) && grid.length >= 1 && grid.length <= 30
    && Array.from(grid).every(row => Array.isArray(row) && row.length >= 1 && row.length <= 30 && row.length === grid[0].length
      && Array.from(row).every(v => Number.isInteger(v) && v >= 0 && v <= 9));
}

export function hasAllValidGrids(grids, queryCount) {
  return Number.isInteger(queryCount) && queryCount > 0 && Array.isArray(grids)
    && grids.length === queryCount && Array.from(grids).every(validGrid);
}
const gridKey = grid => JSON.stringify(grid);
const sourceTier = source => ({ submit_program: 3, verify_demos: 2, cell_transform: 1, d8_conjugate: 1 })[source] ?? 0;

function normalizeCandidate(candidate, queryCount, wave) {
  // Legacy aliases use the same fresh checker / auxiliary callable checks.
  // Keep original provenance while applying the explicit tier taxonomy.
  const aliases = { overflow_program: 'submit_program', transform_alt: 'cell_transform' };
  const source = aliases[candidate.source] ?? candidate.source;
  const tier = candidate.verified === true && hasAllValidGrids(candidate.predictions, queryCount) ? sourceTier(source) : 0;
  return { ...candidate, source, ...(source !== candidate.source ? { original_source: candidate.source } : {}),
    wave: Number.isInteger(wave ?? candidate.wave) && (wave ?? candidate.wave) > 0 ? (wave ?? candidate.wave) : 1,
    tier, verified: tier > 0 };
}
export function candidateRankKey(c) {
  return [c.tier ?? 0, (c.demo_pass_count ?? 0) / Math.max(1, c.demo_count ?? 1),
    -(c.demo_diff_cells ?? 1e9), Number.isFinite(c.robustness) ? c.robustness : -1, c.order ?? 0];
}
function compareDesc(a, b) {
  const ka = candidateRankKey(a), kb = candidateRankKey(b);
  for (let i = 0; i < ka.length; i++) if (ka[i] !== kb[i]) return kb[i] - ka[i];
  return 0;
}
export function rankCandidates(pool, queryCount = Math.max(0, ...(pool ?? []).map(c => c?.predictions?.length ?? 0))) {
  return [...(pool ?? [])].filter(c => Array.isArray(c?.predictions))
    .map(c => normalizeCandidate(c, queryCount)).sort(compareDesc);
}
const provenance = c => c ? { source: c.source, ...(c.original_source ? { original_source: c.original_source } : {}),
  wave: c.wave, tier: c.tier, verified: c.verified, order: c.order ?? null,
  ...(c.coordinate_frame ? { coordinate_frame: c.coordinate_frame, wave_transform: c.wave_transform } : {}),
  demo_pass_count: c.demo_pass_count ?? null } : null;

function chooseAttempts(ranked, queryCount, primaryCandidate = null, fallback = []) {
  const representatives = [], seenWaves = new Set();
  for (const c of ranked) if (c.verified && !seenWaves.has(c.wave)) { representatives.push(c); seenWaves.add(c.wave); }
  const attempts = [], sources = [];
  for (let q = 0; q < queryCount; q++) {
    const first = primaryCandidate && validGrid(primaryCandidate.predictions[q]) ? primaryCandidate
      : representatives.find(c => validGrid(c.predictions[q])) ?? ranked.find(c => validGrid(c.predictions[q]))
        ?? (validGrid(fallback?.[q]) ? { predictions: fallback, source: 'failed_submission', wave: 1, tier: 0, verified: false } : null);
    const attempt1 = first?.predictions[q] ?? null;
    const distinct = c => validGrid(c.predictions[q]) && gridKey(c.predictions[q]) !== gridKey(attempt1);
    const second = representatives.find(c => c.wave !== first?.wave && distinct(c)) ?? ranked.find(distinct);
    attempts.push({ attempt_1: attempt1,
      attempt_1_source: first ? (first === primaryCandidate ? 'formal' : first.source === 'failed_submission' ? 'failed_submission' : `pool:${first.source}`) : null,
      attempt_1_wave: first?.wave ?? null, attempt_1_tier: first?.tier ?? null,
      attempt_2: second ? second.predictions[q] : attempt1,
      attempt_2_source: second ? `pool:${second.source}` : (attempt1 ? 'copy' : null),
      attempt_2_wave: (second ?? first)?.wave ?? null, attempt_2_tier: (second ?? first)?.tier ?? null,
      attempt_2_verified: second ? second.verified : null,
      attempt_2_demo_pass_count: second?.demo_pass_count ?? null, attempt_2_order: second?.order ?? null });
    sources.push({ attempt_1: provenance(first), attempt_2: provenance(second ?? first) });
  }
  return { predictions: attempts.map(a => a.attempt_1), attempts, sources };
}

// Preserve a verified baseline primary while allowing stronger independent
// Wave 3 evidence to replace an unverified or same-wave synthetic backup.
function preserveBaselineAttempts(baseline, later, queryCount, ranked) {
  const selected = structuredClone(baseline);
  const reranked = chooseAttempts(ranked, queryCount);
  for (let q = 0; q < queryCount; q++) {
    const previous = baseline.attempts[q], sources = baseline.sources[q];
    if (!sources.attempt_1?.verified) {
      selected.attempts[q] = reranked.attempts[q];
      selected.sources[q] = reranked.sources[q];
      continue;
    }
    const distinctSecond = validGrid(previous.attempt_2)
      && gridKey(previous.attempt_1) !== gridKey(previous.attempt_2);
    const distinctFromFirst = c => validGrid(c.predictions[q])
      && gridKey(c.predictions[q]) !== gridKey(previous.attempt_1);
    const replacement = later.find(c => c.verified && distinctFromFirst(c))
      ?? (!distinctSecond ? later.find(distinctFromFirst) : null);
    if (!replacement) continue;
    if (distinctSecond && sources.attempt_2?.verified
      && (sources.attempt_1.wave !== sources.attempt_2.wave
        || (sources.attempt_2.tier ?? 0) >= (replacement.tier ?? 0))) continue;
    const secondSource = provenance(replacement);
    selected.sources[q] = { ...sources, attempt_2: secondSource };
    selected.attempts[q] = { ...previous, attempt_2: replacement.predictions[q],
      attempt_2_source: `pool:${replacement.source}`,
      attempt_2_wave: secondSource.wave, attempt_2_tier: secondSource.tier,
      attempt_2_verified: secondSource.verified,
      attempt_2_demo_pass_count: secondSource.demo_pass_count, attempt_2_order: secondSource.order };
  }
  selected.predictions = selected.attempts.map(a => a.attempt_1);
  return selected;
}

export class CandidatePool {
  constructor({ queryCount, candidates = [] } = {}) {
    this.queryCount = queryCount; this.candidates = []; this.keys = new Set(); this.ingest(candidates);
  }
  add(candidate, { wave } = {}) {
    if (!candidate || !Array.isArray(candidate.predictions)) return null;
    const normalized = normalizeCandidate(structuredClone(candidate), this.queryCount ?? candidate.predictions.length, wave);
    // tools.jsonl and live bridge results carry the same receipts. Never dedupe across waves.
    const key = JSON.stringify([normalized.wave, normalized.source, normalized.original_source, normalized.verified,
      normalized.order, normalized.action_id, normalized.demo_pass_count, normalized.demo_diff_cells, normalized.predictions]);
    if (!this.keys.has(key)) { this.keys.add(key); this.candidates.push(normalized); }
    return normalized;
  }
  ingest(candidates, options = {}) { for (const candidate of candidates ?? []) this.add(candidate, options); return this; }
  selectAttempts(queryCount = this.queryCount) {
    const ranked = rankCandidates(this.candidates, queryCount);
    if (!ranked.some(c => c.wave >= 3)) return chooseAttempts(ranked, queryCount);
    return preserveBaselineAttempts(chooseAttempts(ranked.filter(c => c.wave < 3), queryCount),
      ranked.filter(c => c.wave >= 3), queryCount, ranked);
  }
}

// Single-trajectory compatibility. The panel uses CandidatePool for shared waves.
export function selectAttempts({ queryCount, primary = [], fallback = [], pool = [], wave = 1 }) {
  const ranked = rankCandidates(pool, queryCount);
  const matched = ranked.find(c => gridKey(c.predictions) === gridKey(primary));
  const primaryCandidate = primary.length ? (matched ?? normalizeCandidate({ source: 'submit_program',
    verified: true, predictions: primary, wave }, queryCount)) : null;
  return chooseAttempts(ranked, queryCount, primaryCandidate, fallback).attempts;
}
export function submissionEntries(attempts, queryCount) {
  return Array.from({ length: queryCount }, (_, q) => {
    const a = attempts?.[q] ?? {}, first = validGrid(a.attempt_1) ? a.attempt_1 : [[0]];
    return { attempt_1: first, attempt_2: validGrid(a.attempt_2) ? a.attempt_2 : first };
  });
}
