"""Static/CPU evidence audit only: never loads a model or steps a simulator."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003'
STAGE = ROOT / 'code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001'
RAW = ROOT / 'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

def quantile(values, probability):
    values = sorted(values)
    if not values or not 0 <= probability <= 1 or not all(math.isfinite(v) for v in values):
        raise ValueError('nonempty finite values and probability required')
    index = (len(values)-1)*probability
    low = math.floor(index)
    fraction = index-low
    return values[low]*(1-fraction)+values[min(low+1,len(values)-1)]*fraction

def reference(path, symbol):
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    matches = [n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==symbol]
    if not matches:
        raise ValueError(f'missing evidence symbol {path}:{symbol}')
    return {'path':path,'symbol':symbol,'line':matches[0].lineno,'source_sha256':sha(source)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    acceptance = read(STAGE/'13_stage_a_acceptance_receipt.json')
    assert acceptance['selected_search_method']=='MH-CEM' and acceptance['case_count']==960
    config = read(ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/03_validation_execution_config_3080ti.json')
    assert all(sha(ROOT/p)==s for p,s in config['source_sha256'].items())
    inventory = read(STAGE/'12_stage_a_local_file_inventory.json')
    # Inventory source paths are repository-relative, unlike historical TRAIN inventory.
    entries = inventory['files']
    raw_entries = [e for e in entries if '/solve_results/validation/' in e['path']]
    assert len(raw_entries)==960
    rows=[]
    for e in raw_entries:
        p=ROOT/e['path']; assert sha(p)==e['sha256']; rows.append(read(p))
    mh=[r for r in rows if r['method']=='MH-CEM']; assert len(mh)==320
    cases=[]
    for r in mh:
        b=r['outcome']['budget_receipt']; elapsed=b['wall_clock_seconds_diagnostic_only']
        assert elapsed>0 and math.isfinite(elapsed) and b['N_unique_transition_evals']==1024
        cases.append({'anchor':r['sample_id'],'seed':r['seed'],'seconds':elapsed,'unique_transitions':1024,
                      'cache_hits':b['N_cache_hits'],'complete_h4':r['outcome']['complete_sequence_count'],
                      'distinct_scoreable_h4':r['outcome']['h4_scoreable_count'],
                      'unscoreable_completions':r['outcome']['h4_unscoreable_count']})
    times=[c['seconds'] for c in cases]
    full=read(STAGE/'11_formal_stage_a_runtime_receipt.json')
    elapsed=full['actual_elapsed_seconds']; solve_sum=full['in_solve_seconds_sum']; overhead=elapsed-solve_sum
    mh_sum=sum(times); mh_share_elapsed=mh_sum+overhead/3
    yaml_path=ROOT/'code/reference/AirFogSim/examples/config.yaml'
    yaml_source=yaml_path.read_text(encoding='utf-8-sig')
    assert 'simulation_interval: 0.1' in yaml_source and 'traffic_interval: 0.1' in yaml_source
    runtime={'basis':'320 authenticated Stage-A MH-CEM B1024 in-solve timers, RTX3080Ti FP32 batch16',
             'quantile_definition':'linear interpolation at (n-1)*p, no nearest-rank substitution',
             'case_count':320,'median_seconds':quantile(times,.5),'p90_seconds':quantile(times,.9),
             'p95_seconds':quantile(times,.95),'max_seconds':max(times),'in_solve_seconds_sum':mh_sum,
             'unique_transitions_per_in_solve_second':327680/mh_sum,
             'per_case_tps':{'median':quantile([1024/t for t in times],.5),'p90':quantile([1024/t for t in times],.9)},
             'cache_hits':sum(c['cache_hits'] for c in cases),'cache_hit_fraction_of_accounted_requests':
             sum(c['cache_hits'] for c in cases)/(327680+sum(c['cache_hits'] for c in cases)),
             'complete_h4':sum(c['complete_h4'] for c in cases),'distinct_scoreable_h4':sum(c['distinct_scoreable_h4'] for c in cases),
             'unscoreable_completions':sum(c['unscoreable_completions'] for c in cases),
             'matrix_elapsed_seconds':elapsed,'all_method_in_solve_seconds_sum':solve_sum,
             'unattributed_matrix_overhead_seconds':overhead,
             'overhead_note':'Includes runtime preparation/checkpoint loading/fingerprints, orchestration, atomic writes and summary/bootstrap; no separate component timers. Not online end-to-end latency.',
             'simulator_time_grid_seconds':.1,'decision_latency_requirement':'UNRESOLVED',
             'interval_source':{'path':str(yaml_path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(yaml_path),
                  'build_environment':reference('code/scripts/run_p2_single_step_collector_preflight_v1.py','_build_environment'),
                  'preflight_config':reference('code/scripts/small_experiments/airfogsim_strict_dual_graph_preflight.py','build_preflight_config'),
                  'note':'preflight changes max time/traffic/task setup, does not override these intervals. Historical Raw collection records simulation_interval and continuous grid; future closed-loop wall deadline is not set.'},
             'online_B1024_execution_evidence':False,'wall_time_vs_simulated_time':'Different clocks: 0.1s simulated slot is not a frozen wall-clock deadline.',
             'median_ratio_if_realtime_every_0_1s_were_required':quantile(times,.5)/.1,
             'cases':sorted(cases,key=lambda c:(c['anchor'],c['seed']))}
    write('03_runtime_decision_interval_audit.json',runtime)
    src='code/src/pi_jwm/'
    script='code/scripts/'
    env='code/reference/AirFogSim/airfogsim/'
    chain=[]
    def add(stage,status,path,symbol,missing,decision=False):
        chain.append({'stage':stage,'status':status,'evidence':reference(path,symbol),
                      'missing_interface':missing,'researcher_decision_required':decision})
    add('真实 O_t 采集','IMPLEMENTED',script+'run_step2_3_real_airfogsim_raw_contract_finalization_v1.py','_capture','当前raw采集可复用；没有MH闭环调用者。')
    add('History → 当前 state/graph/latent','PARTIAL',src+'step6_1_trained_candidate_rollout_v1.py','prepare_anchor','有posterior初始化；缺history-only在线input builder、固定normalization/slot生命周期及新deadline sidecar。')
    add('离线输入契约不能直接用于实时当前帧','PARTIAL',src+'model_ready_sample_contract_v1.py','build_sample','build_sample要求已完成future_steps；必须独立history-only入口，不补真实未来target。')
    add('当前 CandidateDomain','IMPLEMENTED',src+'step6_3c_candidate_domain_v1.py','from_state','实时context/domain的调用者尚缺，合法集合可为空。')
    add('MH-CEM → H1-H4搜索','IMPLEMENTED',src+'step6_3d_fixed_budget_search_v1.py','solve_fixed_budget','离线solver已有；运行接口只接受已准备anchor。')
    add('冻结World Model一步转移递归','IMPLEMENTED',src+'step6_1_trained_candidate_rollout_v1.py','rollout_one_step_batch','无算法缺口；真实下一轮不能使用这里的预测state。')
    add('Objective与Return边界','IMPLEMENTED',src+'step6_2b_planner_objective_scorer_v1.py','score_candidate_set','实时objective causal side-state组装尚需接线，不改H4。')
    add('可执行winner sequence输出','MISSING',src+'step6_3d_fixed_budget_search_v1.py','SearchOutcome','仅best_objective/fingerprint；动作prefix未作为结果返回。需另行授权不改变排名/预算的输出接口。')
    add('取winner第一步','MISSING',src+'step6_0a_candidate_generation_v1.py','CandidateActionSequence','没有MH winner供steps[0]消费；不能靠指纹还原动作。')
    add('首动作→AirFogSim setters','PARTIAL',src+'step6_0a_candidate_generation_v1.py','compile_candidate','该compile生成模型action tensor，不是仿真命令。缺task/flow/node/UAV索引到真实ID转换和执行前重验。')
    add('真实环境一步执行','IMPLEMENTED',env+'airfogsim_env.py','step','setter已存在；MH四动作族执行桥未接。执行一步按traffic_interval推进，可包含多个simulation_interval。')
    add('真实下一帧独立capture','IMPLEMENTED',script+'collect_step5_5_formal_raw_v1.py','collect_trajectory','已有behavior数据采集循环；非MH闭环证据。')
    add('反馈刷新belief并第二次MH规划','MISSING',script+'run_step6_3d_one_cpu_solve_v1.py','load_frozen_runtime','读取选定离线sample/shard/sidecar，非live history滚动入口；无两轮MH实测。')
    add('NO_SCOREABLE_H4分支','MISSING',src+'step6_3d_fixed_budget_search_v1.py','solve_fixed_budget','best=None可以产生；未定义闭环fallback policy、记录/执行和状态重验。',True)
    add('上一winner余段→MH warm start','MISSING',src+'step6_0a_candidate_generation_v1.py','shift_warm_start','只有shift工具和HybridCompositionBackend接口；MH solver无warm-start参数或注入路径。',True)
    write('01_current_closed_loop_call_chain.json',{'chain':chain,'predicted_state_as_next_real_root':'FORBIDDEN',
          'live_leakage_verdict':'NOT_PROVEN: live input boundary missing; offline contract alone cannot certify a nonexistent loop.',
          'history_only_requirement':'O<=t, A/Y<t only; normalization fixed TRAIN, target/future outcome forbidden.',
          'raw_capture_privileged_audit_fields':'_capture includes internal_metadata (future_task_schedule/future_dag_edges) future task/DAG metadata for audit; strip/whitelist before planner input. Its exclusion flag alone is not a live leakage proof.',
          'hybrid_learned_proposal':'INTERFACE_ONLY_NOT_IMPLEMENTED','search_method':'MH-CEM'})
    fallback=[
        {'option':'现有 rule_fallback 空动作序列 + 显式零CPU/空RB/当前方向speed0 setter组合',
         'evidence':[reference(src+'step6_0a_candidate_generation_v1.py','rule_fallback'),reference(script+'collect_step5_5_formal_raw_v1.py','collect_trajectory')],
         'all_states_constructible':'空序列可构造；真实HOLD需当前UAV观测，缺观测不得猜。未证明所有state可执行。',
         'candidate_domain_contract':'NOT_UNIVERSAL: 有计算eligible base时NOOP被ELIGIBLE_COMP_REQUIRES_TRAIN_ALPHA拒绝；空mob与显式HOLD也不能混同。',
         'simulator_contract':'已有空CPU callback、空无线分配和speed0模式；无MH专用完整桥，不能宣称安全HOLD。',
         'needs_world_model_score':False,'return_birth_blocks_execution':False,'return_birth_blocks_H4_scoring':'仍可能；不得要求fallback先通过H4。',
         'future_target_used':False,'safety_performance_evidence':'只有动作/采集证据；任务超时、能耗和闭环安全未验证。','adopted':False},
        {'option':'当前grammar允许的Comm-NOOP + 合法Comp alpha + 显式PROFILE_HOLD一步',
         'evidence':[reference(src+'step6_3b_candidate_grammar_v1.py','bind_structured_step'),reference(src+'step6_3c_candidate_domain_v1.py','iter_bound')],
         'all_states_constructible':'NO: joint TRAIN支持、compute观测、UAV控制观测和RB50必须满足；domain可为空。',
         'candidate_domain_contract':'逐state bind并检查formal_pool_admitted后才成立；不把任意HOLD看作合法。',
         'simulator_contract':'各primitive有setter；selector/ID桥/资源重验缺失。',
         'needs_world_model_score':False,'return_birth_blocks_execution':False,'return_birth_blocks_H4_scoring':'仍可能',
         'future_target_used':False,'safety_performance_evidence':'无闭环安全/性能证据；只证明现有动作可表示。','adopted':False},
        {'option':'现有采集behavior policy / CpuPolicyAllocator',
         'evidence':[reference(script+'collect_step5_5_formal_raw_v1.py','collect_trajectory'),reference(src+'airfogsim_cpu_policy_v1.py','allocate')],
         'all_states_constructible':'完整behavior依赖真实任务/节点可见性；CPU allocator只是计算分量，不是完整四族policy。',
         'candidate_domain_contract':'完整behavior有offload/Return route，违反Planner v1 Route explicit-NOOP边界；不能直接采用。',
         'simulator_contract':'已有执行路径，不能因此视为当前Planner合法fallback。',
         'needs_world_model_score':False,'return_birth_blocks_execution':False,'return_birth_blocks_H4_scoring':'执行不使用scorer，预测支持限制不消失。',
         'future_target_used':False,'safety_performance_evidence':'生成轨迹不等于闭环安全或baseline胜负证据。','adopted':False}]
    # Bounded existing synthetic TRAIN grammar fixture; no rollout, environment or CUDA.
    sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/tests')]
    from test_step6_3b_candidate_grammar_v1 import fixture
    from pi_jwm.step6_3b_candidate_grammar_v1 import bind_structured_step, StructuredStepChoice, CandidateGrammarViolation
    ctx,dom,state,control,catalog=fixture()
    fallback_oracle=[]
    for comp in ('NOOP','SCALE_0.5','SCALE_0.75','SCALE_1.0'):
        try:
            bound=bind_structured_step(StructuredStepChoice((),comp,'PROFILE_HOLD'),ctx,dom,state,control,catalog)
            fallback_oracle.append({'comp':comp,'admitted':bound.support.formal_pool_admitted,'signature':bound.structural_signature})
        except CandidateGrammarViolation as exc:
            fallback_oracle.append({'comp':comp,'admitted':False,'reason':str(exc)})
    assert fallback_oracle[0]['reason']=='ELIGIBLE_COMP_REQUIRES_TRAIN_ALPHA'
    assert all(x['admitted'] for x in fallback_oracle[1:])
    write('02_fallback_option_audit.json',{'options':fallback,'selection':'AWAITING_RESEARCHER_DECISION',
          'bounded_train_grammar_oracle':fallback_oracle,'oracle_scope':'one CPU synthetic fixture, not all states or simulator safety',
          'requirement':'指定正常无H4分支、domain为空/观测缺失分支、执行重验失败分支；不H3降级、不改H4支持。'})
    # Two transparent planning scenarios, not confidence intervals or measured Stage-B time.
    plans=[{'id':'A','status':'EXISTING_FROZEN_STAGE_B','anchors':64,'seeds':5,'methods':3,'cases':1920,'nominal_transitions':737280,
            'hours_compute_scaled':elapsed*.75/3600,'hours_setup_sensitivity':(solve_sum*.75+overhead*2)/3600,
            'value':'完整三方法budget sensitivity；当前在线预算问题不需要重选方法，额外两方法成本高。'},
           {'id':'B','status':'NEW_RESEARCH_PROPOSAL','anchors':64,'seeds':5,'methods':1,'cases':640,'nominal_transitions':245760,
            'hours_compute_scaled':mh_share_elapsed*.75/3600,'hours_setup_sensitivity':(mh_sum*.75+overhead*(640/960))/3600,
            'value':'MH预算曲线覆盖64锚点；需新runner/provenance和授权，不能冒充原Stage B。'},
           {'id':'C','status':'NEW_RESEARCH_PROPOSAL_NOT_PREREGISTERED','anchors':16,'seeds':3,'methods':1,'cases':96,'nominal_transitions':36864,
            'hours_compute_scaled':mh_share_elapsed*.75*.15/3600,'hours_setup_sensitivity':(mh_sum*.75+overhead*(640/960))*.15/3600,
            'value':'先得到小规模MH预算/延迟曲线；只是建议规模，不冻结subset名单/seed。应保留困难层，匹配既有B1024 cases，避免按新结果挑选；不足支持全部闭环状态。'}]
    bench={}
    for gpu,family in [('4090','pi_jwm_step6_3d_gpu_execution_v1_20260930'),('3080Ti','pi_jwm_step6_3d_3080ti_migration_v1_20260930')]:
        p=ROOT/'code/artifacts/protocols'/family/'04_gpu_batch_throughput.json'
        d=read(p); bench[gpu]={'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p),
                              'native_batch16_tps':next(x['median_transitions_per_second'] for x in d['rows'] if x['batch_size']==16)}
    write('04_stage_b_purpose_cost_comparison.json',{'frozen_purpose':'固定MH-CEM后研究budget与objective、scoreability、candidate diversity、runtime/compute trade-off，为closed-loop online budget提供证据；绝不重新选SEARCH_METHOD。',
          'metrics_requirements':['跨预算同anchor/seed配对objective和六类scoreability，不能将unscoreable填0','distinct action fingerprints及重复率，不把候选数当成语义多样性','in-solve分位数和online准备/编码/执行端到端分开','保留困难锚点/Return边界；subset不用于全矩阵优势宣称'],
          'plans':plans,'recommendation':'先补闭环接口并由研究者确定fallback/latency，再考虑C；C需预注册并另行授权，曲线有用再扩展B。A仅在需要三方法budget敏感性论文证据时值得。',
          'cost_model':'Scenario1 all elapsed scales with transitions. Scenario2 solve scales with transitions, unattributed overhead scales with case count. No Stage-B measurements, not a guaranteed range; seed/anchor complexity, cache, early quota and bootstrap may make scaling nonlinear.',
          'benchmark_evidence':bench,'4090_vs_3080_native_batch16_ratio':bench['4090']['native_batch16_tps']/bench['3080Ti']['native_batch16_tps'],
          '4090_migration':'4090_MIGRATION_NOT_YET_JUSTIFIED',
          '4090_reason':'约15%短native-batch16单anchor吞吐差，不是完整MH-CEM B1024 cpu_cache_and_prefix同身份长期paired实测；不能用batch32推荐或理论规格替代独立qualification。',
          'stage_b':'NOT_STARTED','method_selection_changes':False,'gpu_used':False,'locked_test':False})
    blockers=['BL1 history-only live input/belief/sidecar桥缺失；不能调用需future_steps的离线builder冒充在线输入。',
              'BL2 MH结果缺winner动作序列、first-action输出及真实ID/setter/执行重验桥。',
              'BL3 NO_SCOREABLE_H4和domain为空时fallback未由研究者选择；现有候选非全状态通用。',
              'BL4 decision wall-clock latency requirement未冻结，B1024无在线端到端证据。',
              'BL5 未有两次真实反馈刷新后的MH规划证据；warm-start和learned proposal只有工具/接口。']
    smoke={'authorization':'NOT_EXECUTED_THIS_STEP; separate researcher approval required',
           'minimal_engineering_tasks':['live causal History→input slots/normalization/state/graph/posterior、current-time deadline metadata',
                 '输出winner sequence/first action并以旧discrete结果做不变性证明；需另行授权接口修改',
                 '四族真实ID和单位bridge+原子pre-setter revalidation；Route维持NOOP，不改budget/scorer',
                 '研究者确定fallback、domain-empty stop行为、smoke预算和离线慢速或实时latency模式'],
           'required_trace':['真实capture O_t','history-only重建belief，记录观测时间和posterior来源','MH-CEM frozen参数/H4搜索','winner第一动作或研究者批准fallback',
                 'setter成功回执','one real env.step','独立capture O_t+1','更新History，真实反馈posterior重建','second MH planning cycle'],
           'rejection_gates':['future target poison/drop不改变plan，no target/future outcome依赖','预测rollout state不得写入live root','参数/checkpoint无更新',
                              '无H4、domain-empty、setter失败路径覆盖且不H3降级','实体消失/新任务/Return birth只据真实反馈处理，不伪造slots或扩大支持'],
           'warm_start':'MH integration MISSING; shifting utility exists. Researcher must define accounting, initialization and legal re-evaluation before integration.',
           'learned_proposal':'NOT_IMPLEMENTED; interface location LearnedProposalBackend/HybridCompositionBackend, no implementation authorized.',
           'blockers':blockers}
    write('05_bounded_smoke_requirements.json',smoke)
    private=Path('D:/shen/OB/科研/PIJWM/06策略器与候选动作规划.md')
    acceptance_out={'step':'STEP_6_4A','audit_execution':'PASS','closed_loop_readiness':'BLOCKED','search_method':'MH-CEM',
          'definition_basis':{'path':str(private),'sha256':sha(private),'sections':['1.1','5.1','5.2'],'access':'read-only; no private note copied'},
          'baseline_commit':'afbd7c82bfd549df2e64a5f16be15c1a29741e67','source_semantics_unchanged':True,
          'validation_stage_a_receipt_sha256':sha(STAGE/'13_stage_a_acceptance_receipt.json'),
          'validation_results':960,'stage_b_result_count':0,'stage_b':'NOT_STARTED','gpu':'NOT_USED','locked_test':False,
          'remaining_blockers':blockers,'scope':'Audit accepted; readiness blocked. No scientific algorithm or fallback policy changed.'}
    assert not (RAW/'09_validation_stage_b_budget_diagnostic_receipt.json').exists()
    write('06_go_no_go_receipt.json',acceptance_out)
    write('07_sha_manifest.json',{'files':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for p in sorted(OUT.glob('0[1-6]_*.json'))],
          'source_evidence':[x['evidence'] for x in chain],'frozen_source_sha256':config['source_sha256']})
    print(json.dumps({'readiness':'BLOCKED','mh_runtime_seconds':{k:runtime[k] for k in ['median_seconds','p90_seconds','p95_seconds','max_seconds']},'plans':plans},ensure_ascii=False))

if __name__=='__main__':
    main()
