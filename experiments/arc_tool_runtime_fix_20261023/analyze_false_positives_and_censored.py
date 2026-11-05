import json
from pathlib import Path

gold = json.loads(Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_gr_eval48_kaggle_20261014\data\private\sealed_query_gold.json").read_text(encoding="utf-8"))["tasks"]

runs = {
    "old90": Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_gr48k_pi_medium_kv_g4_20261020\recovery\panel\summary.json"),
    "repro90": Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_gr48k_medium_22_repro_g4_20261023\recovered\panel\summary.json"),
    "s1_65": Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_gr48k_stage1_stage2v3_medium65_g4_20261023\recovered\stage1\medium\panel\summary.json"),
    "s2v3_65": Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_gr48k_stage1_stage2v3_medium65_g4_20261023\recovered\stage2_v3\medium\panel\summary.json"),
    "s2v2_65": Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_stage2_verified_snapshot_dual_g4_20261023\recovered\medium\medium\panel\summary.json"),
    "s2v3_fullfix": Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_tool_runtime_fix_20261023\g4_recovered\stage2_v3_fullfix\medium\panel\summary.json"),
    "s1_fullfix": Path(r"E:\projects\arc_vitarc_ttt_1\experiments\arc_tool_runtime_fix_20261023\g4_recovered\stage1_fullfix\medium\panel\summary.json"),
}

for rname, spath in runs.items():
    if not spath.exists():
        print("MISSING:", rname, spath)
        continue
    s = json.loads(spath.read_text(encoding="utf-8"))
    tasks = s["tasks"]
    elig_correct = []
    elig_wrong = []
    censored = []
    for tid, t in sorted(tasks.items()):
        targets = gold[tid]["outputs"]
        preds = t.get("predictions") or []
        tok = t.get("output_tokens", 0)
        is_elig = t.get("status") == "eligible" or (t.get("answer_source") in ("submit_program", "verified_fallback") and preds)
        if is_elig:
            ok = True
            for qi, target in enumerate(targets):
                if qi >= len(preds) or preds[qi] is None:
                    ok = False
                else:
                    p = preds[qi]
                    cand = p.get("attempt_1") if isinstance(p, dict) else p
                    if cand != target:
                        ok = False
            if ok:
                elig_correct.append((tid, tok))
            else:
                elig_wrong.append((tid, tok))
        else:
            censored.append((tid, tok, t.get("python_calls", 0), t.get("final_checks", 0), t.get("stop_reason")))
    print(f"=== {rname} ===")
    print(f"  Eligible & Correct ({len(elig_correct)}): median tok = {sorted(x[1] for x in elig_correct)[len(elig_correct)//2] if elig_correct else 0}")
    print(f"  Eligible & WRONG   ({len(elig_wrong)}): {elig_wrong}")
    print(f"  Censored/Other     ({len(censored)}): mean tok = {int(sum(x[1] for x in censored)/max(1,len(censored)))}")
