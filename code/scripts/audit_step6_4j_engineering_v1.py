"""Independent audit of r22 CPU engineering receipts; no execution."""
import argparse, json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    rec=json.loads((a.root/'pilot_attempt.json').read_text(encoding='utf-8'))
    assert rec['status']=='ENGINEERING_PASS' and rec['engineering_checks']['env_step_count']==4
    files=sorted(a.root.rglob('decision_*.json'))
    assert len(files)==4
    comm_nonempty=comp_nonempty=outcome_ok=history_ok=0
    for p in files:
        row=json.loads(p.read_text(encoding='utf-8'));assert row['status']=='EXECUTED' and row['environment_step_attempted']
        action=row['action'];hist=row['history_action'];fresh=row['history_outcome']
        assert hist['comm']['empty']==action['comm']['empty'] and hist['comp']['empty']==action['comp']['empty'] and hist['mobility']['empty']==action['mobility']['empty'] and hist['route']['empty']==action['route']['empty']
        assert fresh['capture_method']=='fresh_direct_real_environment_read' and fresh['capture_phase']=='loop_start_decision'
        assert fresh['simulation_time_s']==row['simulation_time_after'] and fresh['frame_index']==row['fresh_observation']['frame_index']
        assert 'slot_transfer_events' in fresh and 'communication_observation' in fresh
        for e in hist['comm']['entries']: assert isinstance(e['task_id'],str) and all(isinstance(x,int) for x in e['rb_indices'])
        for e in hist['comp']['entries']: assert isinstance(e['task_id'],str) and isinstance(e['node_id'],str) and float(e['allocated_cpu_per_s'])>=0
        for e in hist['mobility']['entries']: assert isinstance(e['uav_id'],str) and all(isinstance(float(e[k]),float) for k in ('azimuth_rad','elevation_rad','speed_mps'))
        comm_nonempty += int(bool(hist['comm']['entries']));comp_nonempty += int(bool(hist['comp']['entries']));history_ok += 1;outcome_ok += 1
    assert history_ok==4 and outcome_ok==4
    print(json.dumps({'verdict':'PASS','decision_receipts':4,'env_steps':4,'comm_nonempty_steps':comm_nonempty,'comp_nonempty_steps':comp_nonempty,'history_field_alignment':'PASS','real_outcome_alignment':'PASS'},ensure_ascii=False))
if __name__=='__main__': main()
