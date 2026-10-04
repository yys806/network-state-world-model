"""Planner action eligibility only; never deletes Flow or changes model state."""
from .step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB

COMMUNICABLE_LIFECYCLES=frozenset(('offloading','transmitting'))

def comm_current_task_eligible(*,present,lifecycle,completed=False):
    """Shared exact predicate for tensor-state and current raw-live task rows."""
    return bool(present) and lifecycle in COMMUNICABLE_LIFECYCLES and not bool(completed)

def tensor_comm_task_eligible(state,slot):
    if 'task_lifecycle_index' not in state or 'task_presence' not in state:return False
    if not 0<=slot<state['task_presence'].shape[1]:return False
    index=int(state['task_lifecycle_index'][0,slot])
    lifecycle=LIFECYCLE_VOCAB[index] if 0<=index<len(LIFECYCLE_VOCAB) else None
    completed=bool(state['task_completed'][0,slot]) if 'task_completed' in state else False
    return comm_current_task_eligible(present=state['task_presence'][0,slot],lifecycle=lifecycle,completed=completed)
