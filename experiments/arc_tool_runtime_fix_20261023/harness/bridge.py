"""JSONL transport to the existing persistent workspace and fresh final checker."""
import argparse
import ast
import copy
import json
import signal
from pathlib import Path
import sys
import time

for _depth in (1, 2, 3):
    if len(Path(__file__).resolve().parents) > _depth:
        sys.path.insert(0, str(Path(__file__).resolve().parents[_depth]))
from src.arc_feedback_harness.data import public_task
from src.arc_feedback_harness.feedback import alt_feedback, candidate_feedback, compact_feedback
from src.arc_feedback_harness.workspace import Workspace, Limits, check_candidate
from src.arc_feedback_harness.candidate_source import extract_candidate, executed_definition_prefix

BUDGET_EXHAUSTED = 'This tool budget is exhausted. Submit your complete program.'
STAGNATION_NOTICE = ('This experiment repeats an earlier AST without improved demo results or a changed '
                     'stdout/error outcome. Do not repeat it; test a different hypothesis or call '
                     'submit_program with your complete transform(grid) program.')


class StagnationTracker:
    """Detect recurrences (including alternating cycles) without suppressing execution."""

    def __init__(self):
        self.best_demo_pass_count = 0
        self.stagnation_count = 0
        self.seen = {}
        self.failed_submissions = set()

    def note_progress(self, count):
        self.best_demo_pass_count = max(self.best_demo_pass_count, int(count or 0))

    def observe(self, name, code, execution, *, demo_pass_count=0, eligible=False):
        self.note_progress(demo_pass_count)
        try:
            normalized = ast.dump(ast.parse(code), include_attributes=False)
        except (SyntaxError, TypeError):
            return {}
        # Ignore line numbers, timings, snapshots and diagnostics, which can change
        # solely because of formatting, instrumentation or a prior successful cell.
        outcome = json.dumps({key: execution.get(key) for key in
                              ('ok', 'kind', 'stdout', 'stdout_truncated', 'error')}, sort_keys=True)
        key = (name, normalized)
        previous = self.seen.get(key)
        duplicate_failed = (name == 'submit_program' and not eligible
                            and normalized in self.failed_submissions)
        detected = (name in {'python', 'submit_program'} and previous is not None
                    and previous['outcome'] == outcome
                    and self.best_demo_pass_count <= previous['best']) or duplicate_failed
        self.seen[key] = {'outcome': outcome, 'best': self.best_demo_pass_count}
        if name == 'submit_program' and not eligible:
            self.failed_submissions.add(normalized)
        if detected:
            self.stagnation_count += 1
        return {'stagnation_detected': bool(detected), 'stagnation_count': self.stagnation_count,
                'duplicate_failed_ast': duplicate_failed,
                'best_demo_pass_count': self.best_demo_pass_count}

def model_feedback(value, *, verbose=False):
    """Filter controller-only metadata without altering the full tools log."""
    if isinstance(value, list):
        return [model_feedback(item, verbose=verbose) for item in value]
    if not isinstance(value, dict):
        return value
    hidden = {'verified_snapshot', 'verified_snapshot_saved', 'verified_fallback_saved',
              'verified_fallback_predictions', 'verified_fallback_action_id', 'pool_candidates',
              'executed_prefix_line'}
    if not verbose:
        hidden.update({'repair_diagnosis', 'test_output_summary'})
        if value.get('exploratory_only') is True:
            hidden.add('query_advisories')
    return {key: model_feedback(item, verbose=verbose)
            for key, item in value.items() if key not in hidden}


def recover_submission(task, code, action_id, limits, history, execution):
    """Retry missing workspace dependencies once, keeping only a better check."""
    checks = [*execution.get('demos', []), *execution.get('queries', [])]
    missing_name = any(not result.get('ok') and
                       str(result.get('error', '')).startswith('NameError:')
                       for result in checks)
    if not history or (execution.get('ok') and not missing_name):
        return execution, None
    try:
        stitched_code = extract_candidate(history, code)
    except (SyntaxError, ValueError):
        return execution, None
    if stitched_code == code:
        return execution, None
    stitched = check_candidate(task, stitched_code, f'{action_id}_stitched', limits)
    previous_feedback = candidate_feedback(task, execution)
    stitched_feedback = candidate_feedback(task, stitched)
    more_demos = (stitched_feedback.get('demo_pass_count', 0)
                  > previous_feedback.get('demo_pass_count', 0))
    # A top-level ok only means the checker ran. Every demo/query must finish
    # successfully before calling the retry error-free (semantic misses may remain).
    error_free = (bool(stitched.get('ok'))
                  and len(stitched.get('demos', [])) == len(task['train'])
                  and len(stitched.get('queries', [])) == len(task['test'])
                  and all(result.get('ok') for result in
                          [*stitched.get('demos', []), *stitched.get('queries', [])]))
    if more_demos or error_free:
        return stitched, stitched_code
    return execution, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--trusted-fixture', action='store_true')
    p.add_argument('--verbose-feedback', action='store_true',
                   help='Include repair diagnostics and test summaries in model feedback')
    args = p.parse_args()
    task = public_task(json.loads(Path(args.task).read_text(encoding='utf-8')))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    limits = Limits(trusted_fixture_mode=args.trusted_fixture)
    workspace = None
    history = []
    stagnation = StagnationTracker()
    snapshot = {}
    def save_snapshot(value):
        nonlocal snapshot
        snapshot = copy.deepcopy(value)
        temporary = output / 'verified_fallback.json.tmp'
        temporary.write_text(json.dumps(snapshot), encoding='utf-8')
        temporary.replace(output / 'verified_fallback.json')
        print(json.dumps({**snapshot, 'id': value['action_id']}), flush=True)
    pool_order = 0
    def emit_pool(candidate):
        """Silent second-slot candidate: saved and sent to the controller, never to the model."""
        nonlocal pool_order
        stagnation.note_progress(candidate.get('demo_pass_count', 0))
        pool_order += 1
        record = {**copy.deepcopy(candidate), 'order': pool_order}
        with (output / 'candidates.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps({'event': 'pool_candidate', 'candidate': record}), flush=True)
    def terminate(_signum, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    try:
        for line in sys.stdin:
            req = json.loads(line)
            if req['name'] == 'close':
                break
            start = time.monotonic()
            candidate_predictions = []
            demo_pass_count = 0
            demo_diff_cells = 0
            final_check_consumed = False
            extracted_code = None
            extraction_error = None
            execution_name = req['name']
            if execution_name == 'python_candidate':
                try:
                    extracted_code = extract_candidate(history, req['code'])
                    execution_name = 'submit_program'
                except (SyntaxError, ValueError) as exc:
                    execution_name = 'candidate_extraction_failed'
                    extraction_error = str(exc)
            if req['name'] == 'python':
                if workspace is None:
                    workspace = Workspace(task, limits, on_verified=save_snapshot, on_candidate=emit_pool)
                execution = workspace.run(req['code'], req['id'])
                if execution.get('ok'):
                    history.append(req['code'])
                elif execution.get('executed_prefix_line') is not None:
                    prefix = executed_definition_prefix(req['code'], execution['executed_prefix_line'])
                    if prefix:
                        history.append(prefix)
                for report in execution.get('verification_reports', []):
                    stagnation.note_progress(sum(d.get('kind') == 'pass' for d in report.get('demos', [])))
                if workspace.closed:
                    workspace = None
                    history.clear()
                    execution = {**execution, 'workspace_reset': True,
                        'recovery': 'The Python workspace was stopped. Its variables are lost. '
                                    'Your next Python call starts a fresh workspace with the public task data. '
                                    'Previous code will not be replayed.'}
                feedback = {k: v for k, v in execution.items() if not k.startswith('verified_fallback_')}
                predictions = []; eligible = False
            elif execution_name == 'submit_program':
                final_check_consumed = True
                execution = check_candidate(task, extracted_code or req['code'], req['id'], limits)
                if extracted_code is None and history:
                    execution, extracted_code = recover_submission(
                        task, req['code'], req['id'], limits, history, execution)
                feedback = candidate_feedback(task, execution)
                eligible = feedback['eligible']
                predictions = [q['grid'] for q in execution['queries']] if eligible else []
                demos = feedback.get('demos', [])
                demo_pass_count = int(feedback.get('demo_pass_count', 0))
                demo_diff_cells = sum(int(d.get('different_cells', 999)) for d in demos if d.get('kind') != 'pass')
                raw_queries = execution.get('queries', [])
                query_statuses = feedback.get('queries', [])
                aligned_queries = (
                    bool(execution.get('ok'))
                    and len(raw_queries) == len(task['test'])
                    and len(query_statuses) == len(task['test'])
                )
                partial_predictions = (
                    [copy.deepcopy(r['grid']) if s.get('valid') else None
                     for r, s in zip(raw_queries, query_statuses)]
                    if aligned_queries else []
                )
                has_any_valid_query = any(grid is not None for grid in partial_predictions)
                candidate_predictions = partial_predictions if has_any_valid_query else []
                keep_candidates = req['name'] != 'python_candidate' or eligible
                if has_any_valid_query and keep_candidates:
                    emit_pool({'source': 'overflow_program' if req['name'] == 'python_candidate' else 'submit_program',
                               'verified': bool(eligible), 'demo_pass_count': demo_pass_count,
                               'demo_count': len(task['train']), 'demo_diff_cells': demo_diff_cells,
                               'predictions': candidate_predictions, 'action_id': req['id']})
                alt = alt_feedback(task, execution, candidate_predictions)
                if alt is not None and keep_candidates:
                    feedback = {**feedback, 'transform_alt': alt['summary']}
                    if alt['predictions']:
                        emit_pool({'source': 'transform_alt', 'verified': alt['summary'].get('all_demos_pass', False),
                                   'demo_pass_count': alt['summary'].get('demo_pass_count', 0),
                                   'demo_count': len(task['train']), 'demo_diff_cells': alt['demo_diff_cells'],
                                   'predictions': alt['predictions'], 'action_id': req['id']})
                if keep_candidates and execution.get('d8_conjugates'):
                    for conj_preds in execution['d8_conjugates']:
                        emit_pool({'source': 'd8_conjugate', 'verified': True,
                                   'demo_pass_count': len(task['train']),
                                   'demo_count': len(task['train']), 'demo_diff_cells': 0,
                                   'predictions': conj_preds, 'action_id': req['id']})
            elif execution_name == 'candidate_extraction_failed':
                execution = {'ok': False, 'kind': 'candidate_extraction',
                             'error': extraction_error}
                feedback = execution; predictions = []; eligible = False
            else:
                raise ValueError('Unknown tool')
            # Overflow probes are not explicit submissions. Only a verified
            # program is accepted; failed probes leave all formal checks intact.
            if req['name'] == 'python_candidate':
                final_check_consumed = bool(eligible)
                if not eligible:
                    feedback = BUDGET_EXHAUSTED
                    candidate_predictions = []
                    execution_name = 'python_budget_blocked'
            stagnation_status = stagnation.observe(req['name'], req['code'], execution,
                demo_pass_count=demo_pass_count, eligible=eligible)
            if stagnation_status.get('stagnation_detected'):
                feedback = {**feedback, 'stagnation_notice': STAGNATION_NOTICE}
            repairable_submission_failure = (
                execution_name == 'submit_program'
                and not eligible
                and (
                    feedback.get('kind') in {'syntax', 'contract', 'runtime'}
                    or (
                        feedback.get('kind') == 'failed_check'
                        and (
                            any(d.get('kind') != 'pass' for d in feedback.get('demos', []))
                            or any(q.get('valid') is False for q in feedback.get('queries', []))
                        )
                    )
                )
            )
            record = {**req, 'execution': execution, 'feedback': feedback,
                      'eligible': eligible, 'predictions': predictions,
                      'candidate_predictions': candidate_predictions,
                      'demo_pass_count': demo_pass_count, 'demo_diff_cells': demo_diff_cells,
                      'executed_tool': execution_name, 'extracted_code': extracted_code,
                      'final_check_consumed': final_check_consumed,
                      'repairable_submission_failure': repairable_submission_failure,
                      **stagnation_status}
            with (output / 'tools.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            result = {'id': req['id'], 'feedback': compact_feedback(model_feedback(
                          feedback, verbose=args.verbose_feedback)),
                      'is_error': execution_name == 'python_budget_blocked' or not execution.get('ok', False), 'eligible': eligible,
                      'predictions': predictions, 'candidate_predictions': candidate_predictions,
                      'demo_pass_count': demo_pass_count, 'demo_diff_cells': demo_diff_cells,
                      'final_check_consumed': final_check_consumed, 'executed_tool': execution_name,
                      'repairable_submission_failure': repairable_submission_failure,
                      'verified_fallback_predictions': copy.deepcopy(snapshot.get('verified_fallback_predictions', [])),
                      'verified_fallback_action_id': snapshot.get('action_id'),
                      'verified_snapshot': copy.deepcopy(snapshot.get('verified_snapshot')),
                      'seconds': time.monotonic() - start, **stagnation_status}
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if workspace is not None:
            workspace.close()


if __name__ == '__main__':
    main()
