import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';

// One sequential JSONL transport per task. A terminal process is never reused.
export function makeProcessBridge({ command, args, stderr, signal }) {
  let child, terminal, exit, closePromise, verifiedSnapshot = null, queue = [];
  const pool = []; // silent second-slot candidates, kept even if the workspace dies later
  const fail = error => {
    terminal ??= error;
    for (const pending of queue.splice(0)) pending.reject(terminal);
  };
  const abort = () => {
    fail(new Error(`workspace_aborted: ${signal.reason}`));
    child?.kill('SIGTERM');
  };
  async function waitExit(ms) {
    let timer;
    try { return await Promise.race([exit.then(() => true),
      new Promise(resolve => { timer = setTimeout(() => resolve(false), ms); })]); }
    finally { clearTimeout(timer); }
  }
  return {
    get verifiedSnapshot() { return verifiedSnapshot && structuredClone(verifiedSnapshot); },
    get poolCandidates() { return structuredClone(pool); },
    call(message) {
      if (signal?.aborted) abort();
      if (terminal) return Promise.reject(terminal);
      if (!child) {
        child = spawn(command, args, { stdio: ['pipe', 'pipe', 'pipe'] });
        if (stderr) child.stderr.pipe(stderr);
        else child.stderr.resume();
        exit = new Promise(resolve => child.once('close', resolve));
        child.on('error', fail);
        child.stdin.on('error', fail);
        child.on('exit', (code, reason) => fail(new Error(`workspace_exit_${reason ?? code}`)));
        createInterface({ input: child.stdout }).on('line', line => {
          const pending = queue[0];
          try {
            const message = JSON.parse(line);
            if (message.event === 'pool_candidate') {
              if (message.candidate) pool.push(message.candidate);
              return;
            }
            if (message.event === 'verified_candidate') {
              verifiedSnapshot = structuredClone(message);
              return;
            }
            if (!pending) return;
            queue.shift();
            pending.resolve(message);
          }
          catch (error) { pending?.reject(error); fail(error); child.kill('SIGTERM'); }
        });
        signal?.addEventListener('abort', abort, { once: true });
      }
      return new Promise((resolve, reject) => {
        queue.push({ resolve, reject });
        child.stdin.write(JSON.stringify(message) + '\n', error => { if (error) fail(error); });
      });
    },
    close() {
      // solveTask and its panel both own cleanup. Share one bounded drain so a
      // deadline cannot pay the three-second termination sequence twice.
      return closePromise ??= (async () => {
        signal?.removeEventListener('abort', abort);
        fail(new Error('workspace_closed'));
        if (!child) { stderr?.end(); return; }
        if (!child.stdin.destroyed) child.stdin.end();
        if (await waitExit(1000)) return;
        child.kill('SIGTERM');
        if (await waitExit(1000)) return;
        child.kill('SIGKILL');
        await waitExit(1000);
      })();
    }
  };
}
