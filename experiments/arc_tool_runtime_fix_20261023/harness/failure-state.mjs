export function isBackendFailure(error) {
  return /EngineCore|EngineDead|CUDA.*out of memory|(?:^|\s)50[0234](?:\D|$)|tokenize_http_5|ECONN|EPIPE|fetch failed|connection error|socket hang up/i.test(String(error));
}

export function taskCounts(records, running, total) {
  const finished = Object.values(records), active = Object.values(running);
  const started = [...finished, ...active].filter(x => (x.requests ?? 0) > 0).length;
  return {
    completed_tasks: finished.filter(x => ['eligible', 'censored'].includes(x.status)).length,
    terminal_records: finished.length,
    started_tasks: started,
    unstarted_tasks: total - started,
    eligible_tasks: finished.filter(x => x.status === 'eligible' || x.verified_fallback_used).length,
    submitted_program_tasks: finished.filter(x => x.status === 'eligible').length,
    verified_fallback_tasks: finished.filter(x => x.verified_fallback_used).length,
    fallback_tasks: finished.filter(x => !x.predictions?.length && Array.isArray(x.fallback_predictions) && x.fallback_predictions.length > 0).length,
    interrupted_tasks: finished.filter(x => x.status === 'engine_interrupted').length,
    task_errors: finished.filter(x => x.status === 'task_error').length,
  };
}
