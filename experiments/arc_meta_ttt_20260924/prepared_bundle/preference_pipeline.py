"""Conditional, matched continued-SFT versus DPO+chosen-NLL experiment.

This stage never launches a GPU allocation and never promotes a Kaggle model.
Its parent must first validate stage1. All phases share an absolute deadline.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def lines(path):
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def frozen_views(directory):
    from score_gold import load_frozen_views
    manifest,_,views=load_frozen_views(Path(directory)/'manifest.json')
    return manifest,{(v['query_id'],v['view_id']):v for v in views}

def model_fingerprint(path):
    from score_gold import checkpoint_identity
    return checkpoint_identity(path)[0]

def choose_training(proposals,selected_ids,max_queries=256):
    """One frozen hash-selected view/query; up to three negatives for exact R1 scoring."""
    groups=defaultdict(lambda:defaultdict(list))
    for p in proposals:
        if p.get('split')!='train' or p.get('training_eligible') is not True:
            raise ValueError('Nontraining proposal in training file')
        if p['task_id'] not in selected_ids:
            continue
        if len(p['prompt_tokens'])+len(p['chosen_tokens'])>4096:
            continue
        groups[p['query_id']][p['view_id']].append(p)
    rank={task:index for index,task in enumerate(selected_ids)}
    pools=[]
    for q,views in groups.items():
        view=min(views,key=lambda v:hashlib.sha256(('preference-view:'+q+':'+v).encode()).hexdigest())
        pool=sorted(views[view],key=lambda p:len(p['edits']))
        if len({p['pair_id'] for p in pool})!=len(pool) or not 1<=len(pool)<=3:
            raise ValueError('Invalid proposal pool')
        pools.append(pool)
    # selected.jsonl already preserves frozen baseline weakness/family-roundrobin.
    chosen=sorted(pools,key=lambda pool:rank[pool[0]['task_id']])[:max_queries]
    if len(chosen)<32:
        raise ValueError('Fewer than32 eligible unique chosen episodes')
    return chosen

def reference_record(pair,role,logprobs,artifact):
    tokens=pair[role+'_tokens']
    return {'pair_id':pair['pair_id'],'binding':pair['binding'],'role':role,
            'tokens':tokens,'reference_artifact':artifact,
            'conditioning':'actual_response_prefix','eval_mode':True,
            'scoring_artifact_sha256':hashlib.sha256(canonical({'pair_id':pair['pair_id'],'role':role,'tokens':tokens,'target_logprobs':logprobs,'artifact':artifact})).hexdigest(),
            'target_logprobs':logprobs}

def cache_phase(args):
    from score_gold import score_batch, make_proposals
    from preference_pairs import attach_reference_scores
    import torch
    from transformers import AutoModelForCausalLM
    start=time.time();root=Path(args.root);out=root/'results/preference'
    manifest,views=frozen_views(args.materialized_dir)
    selected=[r['task_id'] for r in lines(args.selected_training)]
    pools=choose_training(lines(args.proposals),selected)
    model=AutoModelForCausalLM.from_pretrained(args.base_model,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    checkpoint=model_fingerprint(args.base_model)
    scoring_times=[];token_work=0
    def scored(view):
        nonlocal token_work
        if time.time()+30>=args.deadline:raise TimeoutError('Reference cache deadline')
        began=time.time()
        with torch.inference_mode():
            result=score_batch(model,[view],'cuda:0')[0]
        scoring_times.append(time.time()-began)
        token_work+=len(view['prompt_tokens'])+len(view['answer_tokens'])
        return result
    def exact_pool(pool):
        first=pool[0]
        if any(p['prompt_tokens']!=first['prompt_tokens'] or p['chosen_tokens']!=first['chosen_tokens'] for p in pool):
            raise ValueError('Contrast pool has inconsistent chosen context/answer')
        artifact={'stage':'merged_stage1_sft','checkpoint_sha256':checkpoint,
                  **{k:first['proposer_artifact'][k] for k in ('tokenizer_sha256','serialization_sha256','scoring_contract_sha256')}}
        baseview={'prompt_tokens':first['prompt_tokens'],'answer_tokens':first['chosen_tokens']}
        chosen=scored(baseview)
        rows=[]
        for pair in pool:
            rejected=scored({'prompt_tokens':pair['prompt_tokens'],'answer_tokens':pair['rejected_tokens']})
            rows.append(attach_reference_scores(pair,
                reference_record(pair,'chosen',chosen['target_logprobs'],artifact),
                reference_record(pair,'rejected',rejected['target_logprobs'],artifact),expected_reference_artifact=artifact))
        return rows
    training=[]
    for pool in pools:
        scores=exact_pool(pool)
        training.append(max(scores,key=lambda p:(sum(p['reference_scores']['rejected']['target_logprobs']),p['pair_id'])))
    write(out/'training_cache.json',training)
    families=defaultdict(set)
    for q in manifest['queries']:
        if q['split']=='holdout':families[q['family_id']].add(q['query_id'])
    if len(families)!=50:raise ValueError('Require all50 frozen holdout families')
    evaluation=[]
    # Same fixed1episode/family x8views; proposals created under R1 only once.
    for family in sorted(families):
        q=min(families[family],key=lambda key:hashlib.sha256(('margin:'+key).encode()).hexdigest())
        selected_views=[v for (qid,_),v in views.items() if qid==q]
        if len(selected_views)!=8:raise ValueError('Holdout margin views incomplete')
        for view in selected_views:
            gold=scored(view)
            original_base=pools[0][0]['proposer_artifact']['original_base_sha256']
            proposals=make_proposals(view,gold['full_logprobs'],checkpoint,original_base,manifest)
            for p in proposals:
                p.update(query_id=q,task_id=view['task_id'],family_id=family,split='holdout',view_id=view['view_id'],training_eligible=False)
            evaluation.extend(exact_pool(proposals))
    write(out/'holdout_contrast_cache.json',evaluation)
    write(out/'cache_status.json',{'complete':True,'training_queries':len(training),'holdout_families':50,
        'holdout_views':400,'holdout_pairs':len(evaluation),'model_sha256':checkpoint,
        'elapsed_seconds':time.time()-start,'scored_tokens':token_work,'forward_seconds':sum(scoring_times),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})

def logprobs(model,prompt,answer):
    import torch
    ids=torch.tensor([prompt+answer],device='cuda')
    logits=model(input_ids=ids,use_cache=False).logits[0,len(prompt)-1:-1].float()
    return torch.log_softmax(logits,dim=-1).gather(-1,ids[0,len(prompt):,None]).squeeze(-1)

def preference_loss(chosen_sum,rejected_sum,reference_chosen,reference_rejected,length,objective):
    import torch.nn.functional as F
    nll=-chosen_sum/length
    if objective=='sft':return nll
    if objective!='pairwise':raise ValueError(objective)
    return F.softplus(-0.1*((chosen_sum-reference_chosen)-(rejected_sum-reference_rejected)))+nll

def train_phase(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import LoraConfig,get_peft_model
    from train_grid_adapter import TARGET_MODULES,validate_tokenizer
    from preference_pairs import _validate_pair, fingerprint
    cache=read(Path(args.root)/'results/preference/training_cache.json')
    freeze_path=Path(args.root)/'results/preference/TRAINING_FREEZE.json'
    if not args.timing_probe:
        freeze=read(freeze_path)
        if freeze['cache_sha256']!=sha(freeze_path.parent/'training_cache.json') or freeze['updates']!=args.updates:
            raise ValueError('Training cache/update count differs from frozen comparison')
    expected_model=model_fingerprint(args.base_model)
    for pair in cache:
        _validate_pair(pair)
        if (pair.get('split')!='train' or pair.get('training_eligible') is not True
                or pair.get('reference_scores_complete') is not True
                or pair.get('reference_cache_sha256')!=fingerprint(pair['reference_scores'])
                or pair['reference_scores']['binding']!=pair['binding']
                or pair['reference_scores']['pair_id']!=pair['pair_id']
                or pair['reference_scores']['reference_artifact']['checkpoint_sha256']!=expected_model):
            raise ValueError('Training cache reference/model/provenance mismatch')
    selected=cache[:args.updates*8]
    if len(selected)!=args.updates*8:raise ValueError('Requested updates exceed one pass')
    set_seed(20260922)
    base=AutoModelForCausalLM.from_pretrained(args.base_model,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    from score_gold import load_tokenizer
    tokenizer=load_tokenizer(args.base_model)
    validate_tokenizer(tokenizer,base)
    model=get_peft_model(base,LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=TARGET_MODULES))
    if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise ValueError('NonLoRA trainable')
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-6,weight_decay=0)
    model.train();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    timings=[];records=[]
    for step in range(args.updates):
        if time.time()+max(timings,default=30)+90>=args.deadline:raise TimeoutError('Insufficient time for complete matched update count')
        started=time.time();optimizer.zero_grad(set_to_none=True)
        group=selected[step*8:(step+1)*8]
        for pair in group:
            # Match positive-response dropout across both objectives.
            set_seed(int(hashlib.sha256(('dropout:'+pair['pair_id']).encode()).hexdigest()[:8],16))
            chosen=logprobs(model,pair['prompt_tokens'],pair['chosen_tokens']).sum()
            rejected=logprobs(model,pair['prompt_tokens'],pair['rejected_tokens']).sum() if args.objective=='pairwise' else chosen.detach()
            refs=pair['reference_scores']
            loss=preference_loss(chosen,rejected,sum(refs['chosen']['target_logprobs']),sum(refs['rejected']['target_logprobs']),len(pair['chosen_tokens']),args.objective)
            if not torch.isfinite(loss):raise ValueError('Nonfinite preference loss')
            (loss/len(group)).backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or (step==0 and float(norm)<=0):raise ValueError('Invalid preference gradient')
        optimizer.step();timings.append(time.time()-started)
        records.append({'update':step+1,'seconds':timings[-1],'loss_last':float(loss.detach()),'grad_norm':float(norm)})
        write(out/'progress.json',{'objective':args.objective,'updates':records,'planned_updates':args.updates})
        print('PREFERENCE_UPDATE',args.objective,step+1,timings[-1],flush=True)
    model.save_pretrained(out/'adapter',safe_serialization=True);tokenizer.save_pretrained(out/'adapter')
    write(out/'training_status.json',{'complete':True,'objective':args.objective,'updates':args.updates,
        'pair_ids':[p['pair_id'] for p in selected],'cache_sha256':sha(Path(args.root)/'results/preference/training_cache.json'),
        'update_seconds':timings,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})

def margin_phase(args):
    import torch
    from transformers import AutoModelForCausalLM
    from score_gold import score_batch
    cache_path=Path(args.root)/'results/preference/holdout_contrast_cache.json'
    if sha(cache_path)!=read(cache_path.parent/'TRAINING_FREEZE.json')['holdout_pool_sha256']:
        raise ValueError('Margin pool differs from frozen comparison')
    cache=read(cache_path)
    groups=defaultdict(list)
    for pair in cache:groups[(pair['query_id'],pair['view_id'])].append(pair)
    model=AutoModelForCausalLM.from_pretrained(args.base_model,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    records=[]
    with torch.inference_mode():
        for key,pool in sorted(groups.items()):
            if time.time()+20>=args.deadline:raise TimeoutError('Incomplete margin evaluation')
            first=pool[0]
            prompts=[{'prompt_tokens':first['prompt_tokens'],'answer_tokens':first['chosen_tokens']}]+[
                {'prompt_tokens':p['prompt_tokens'],'answer_tokens':p['rejected_tokens']} for p in pool]
            values=[score_batch(model,[view],'cuda:0')[0] for view in prompts]
            gold=-values[0]['sequence_nll'];wrong=max(-v['sequence_nll'] for v in values[1:])
            r_gold=sum(first['reference_scores']['chosen']['target_logprobs'])
            r_wrong=max(sum(p['reference_scores']['rejected']['target_logprobs']) for p in pool)
            records.append({'query_id':key[0],'view_id':key[1],'family_id':first['family_id'],
                'margin':gold-wrong,'reference_margin':r_gold-r_wrong,'pool_ids':[p['pair_id'] for p in pool]})
    if len(records)!=400:raise ValueError('Margin evaluation incomplete')
    write(args.output,records)

def paired_margin(rows,other=None):
    keys=[(r['query_id'],r['view_id']) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate margin query/view')
    other_map={(r['query_id'],r['view_id']):r for r in other} if other else None
    if other is not None and (len(other_map)!=len(other) or set(other_map)!=set(keys)):
        raise ValueError('Margin arm query/view mismatch')
    families=defaultdict(list)
    for row in rows:
        if other_map is not None:
            matching=other_map[(row['query_id'],row['view_id'])]
            if row['family_id']!=matching['family_id'] or row.get('pool_ids')!=matching.get('pool_ids'):
                raise ValueError('Margin arm family/contrast pool mismatch')
        ref=row['reference_margin'] if other_map is None else other_map[(row['query_id'],row['view_id'])]['margin']
        families[row['family_id']].append(row['margin']-ref)
    if len(families)!=50 or any(len(v)!=8 for v in families.values()):raise ValueError('Incomplete margin families/views')
    delta=[sum(v)/len(v) for _,v in sorted(families.items())]
    rng=random.Random(20260922)
    draws=sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(10000))
    return {'mean_delta':sum(delta)/len(delta),'one_sided_95_lower':draws[499],'pass':draws[499]>0,'families':50}

def run_command(command,log,deadline):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('Preference stage deadline')
    with log.open('w',encoding='utf8') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
        try:
            if p.wait(timeout=remaining):raise RuntimeError('Preference subprocess failed: '+str(log))
        except BaseException:
            if p.poll() is None:
                if os.name=='posix':os.killpg(p.pid,signal.SIGKILL)
                else:p.kill()
                p.wait(timeout=30)
            raise

def orchestrate(args):
    root=Path(args.root);out=root/'results/preference';out.mkdir(parents=True,exist_ok=False)
    deadline=min(args.deadline,time.time()+6600)
    state={'state':'preflight','deployment_ready':False,'deadline':deadline,'started':time.time()}
    write(out/'status.json',state)
    shared=['--root',str(root),'--base-model',args.base_model,'--materialized-dir',args.materialized_dir,
            '--selected-training',args.selected_training,'--proposals',args.proposals]
    def invoke(mode,extra,seconds=None):
        end=min(deadline,time.time()+seconds) if seconds else deadline
        run_command([sys.executable,'-u',str(Path(__file__).resolve()),'--mode',mode,*shared,*extra,'--deadline',str(end)],out/(mode+'_'+str(time.time_ns())+'.log'),end)
    try:
        invoke('cache',[],1200)
        cache=read(out/'training_cache.json');cs=read(out/'cache_status.json')
        # Benchmark a COMPLETE scratch optimizer update on longest responses.
        # It is discarded; both controlled arms restart from exactly R1.
        original=cache[:]
        ordered=sorted(cache,key=lambda p:len(p['prompt_tokens'])+len(p['chosen_tokens']),reverse=True)
        write(out/'training_cache.json',ordered)
        invoke('train',['--objective','pairwise','--updates','1','--output',str(out/'timing_probe'),'--timing-probe'],600)
        timing=read(out/'timing_probe/training_status.json')['update_seconds'][0]
        write(out/'training_cache.json',original)
        manifest,_=frozen_views(args.materialized_dir)
        # Conservative token-throughput forecast from exact R1 reference forwards.
        gold_tokens=sum(sum(q.get('sequence_tokens',[])) for q in manifest['queries'] if q['split'] in ('holdout','public'))
        if not gold_tokens:
            _,vm=frozen_views(args.materialized_dir)
            gold_tokens=sum(len(v['prompt_tokens'])+len(v['answer_tokens']) for v in vm.values() if v['split'] in ('public','holdout'))
        forward_rate=cs['forward_seconds']/max(cs['scored_tokens'],1)
        # Two fullgold tests + two complete contrast tests + merge/load/export reserve.
        eval_reserve=1.5*(2*gold_tokens*forward_rate+2*cs['forward_seconds'])+900
        update_budget=deadline-time.time()-eval_reserve-300
        updates=min(100,len(cache)//8,int(update_budget/max(2.5*timing,1)))
        if updates<1:
            state.update(state='skipped_insufficient_measured_time',evaluation_reserve_seconds=eval_reserve)
            return
        write(out/'TRAINING_FREEZE.json',{'updates':updates,'cache_sha256':sha(out/'training_cache.json'),
              'holdout_pool_sha256':sha(out/'holdout_contrast_cache.json'),'objective_pairwise':'DPO beta0.1 plus meanchosenNLL alpha1',
              'objective_control':'meanchosenNLL','learning_rate':1e-6,'probe_longest_update_seconds':timing,
              'evaluation_reserve_seconds':eval_reserve,'created':time.time()})
        for arm in ('sft','pairwise'):
            state.update(state='training',arm=arm,updates=updates);write(out/'status.json',state)
            train=root/'results/training'/('preference_'+arm)
            invoke('train',['--objective',arm,'--updates',str(updates),'--output',str(train)],max(120,deadline-time.time()-eval_reserve/2))
            model=root/'results/models'/('preference_'+arm)
            run_command([sys.executable,str(root/'merge_stage1.py'),'--base',args.base_model,'--adapter',str(train/'adapter'),
                '--output',str(model),'--report',str(out/(arm+'_merge.json')),'--train-jsonl',args.selected_training,
                '--excluded-ids',str(root/'data/excluded_ids.json')],out/(arm+'_merge.log'),deadline)
            exports=read(root/'results/model_exports.json');exports=exports['models'] if isinstance(exports,dict) else exports
            exports.append({'name':'preference_'+arm,'path':model.relative_to(root).as_posix(),'ready':True,
                            'integrity_report_sha256':sha(out/(arm+'_merge.json'))});write(root/'results/model_exports.json',exports)
            run_command([sys.executable,str(root/'score_gold.py'),'score','--manifest',str(Path(args.materialized_dir)/'manifest.json'),
                '--model-path',str(model),'--arm','candidate','--out-dir',str(out/(arm+'_gold')),'--splits','public,holdout',
                '--max-seconds',str(max(1,deadline-time.time()))],out/(arm+'_gold.log'),deadline)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--mode','margin',*shared,
                '--base-model',str(model),'--output',str(out/(arm+'_margins.json')),'--deadline',str(deadline)]
            run_command(command,out/(arm+'_margin.log'),deadline)
        from loss_contract import aggregate_gold_nll
        reference_path=Path(args.reference_gold)
        reference=[{**r,'arm':'base'} for r in lines(reference_path)]
        loss_reports={}
        goldmanifest=[q for q in manifest['queries'] if q['split'] in ('public','holdout')]
        for arm in ('sft','pairwise'):
            loss_reports[arm]=aggregate_gold_nll(goldmanifest,reference+list(lines(out/(arm+'_gold')/'score_rows.jsonl')))
        pw=read(out/'pairwise_margins.json');control=read(out/'sft_margins.json')
        margin_ref=paired_margin(pw);margin_control=paired_margin(pw,control)
        nonincrease={split:loss_reports['pairwise']['splits'][split]['delta_token_nll']<=0
                     for split in ('public','holdout')}
        passes=all(nonincrease.values()) and margin_ref['pass'] and margin_control['pass']
        write(out/'comparison.json',{'loss_reports':loss_reports,'pairwise_margin_vs_R1':margin_ref,
              'pairwise_margin_vs_continued_sft':margin_control,'chosen_nll_nonincrease_vs_R1':nonincrease,
              'research_gate_pass':passes,'solver_panel_completed':False,
              'deployment_ready':False,'qualification':'Conditional development loss/margin experiment only'})
        state.update(state='completed',research_gate_pass=passes,deployment_ready=False)
    except Exception as exc:
        state.update(state='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['finished']=time.time();write(out/'status.json',state)
        write(out/'FREEZE.json',{'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='FREEZE.json' and not p.name.endswith('.tmp')}})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['run','cache','train','margin'],default='run')
    for name in ('root','base-model','materialized-dir','selected-training','proposals'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--reference-gold');p.add_argument('--output');p.add_argument('--objective',choices=['sft','pairwise'])
    p.add_argument('--updates',type=int,default=1)
    p.add_argument('--timing-probe',action='store_true')
    args=p.parse_args()
    if args.mode=='run' and not args.reference_gold:raise ValueError('--reference-gold required')
    {'run':orchestrate,'cache':cache_phase,'train':train_phase,'margin':margin_phase}[args.mode](args)

if __name__=='__main__':main()
