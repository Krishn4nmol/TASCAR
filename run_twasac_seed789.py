import config
config.RANDOM_SEED = 42
import numpy as np, torch
from collections import Counter
from simulator import AzureDataLoader
from scache import SCache
from transformer_encoder import TransformerEncoder, StateHistoryBuffer
from sac_agent import SACAgent
from train_tascar import normalize_state
from config import NUM_QUEUES, TASCAR_DELTA, TRANSFORMER_DIM, NUM_FUNCTIONS, EVAL_CALLS

print("Loading workload (100k calls)...")
loader = AzureDataLoader()
day1 = loader.load_day(1)
fc = Counter(c.function_id for c in day1)
top = set(f for f,_ in fc.most_common(NUM_FUNCTIONS))
workload = [c for c in day1 if c.function_id in top][:EVAL_CALLS]
print(f"  {len(workload)} calls loaded")

state_dim = NUM_QUEUES * 7
enc = TransformerEncoder(state_dim)
agent = SACAgent(transformer_dim=TRANSFORMER_DIM, action_dim=27, transformer=enc)
agent.load('trained_model_tascar_seed5/best/')

scache = SCache()
history = StateHistoryBuffer(10, state_dim)
history.add(normalize_state(scache.get_state()))
total = cold = calls = 0

for call in workload:
    is_warm = scache.handle_request(call)
    total += 1
    if not is_warm: cold += 1
    calls += 1
    if calls % TASCAR_DELTA == 0:
        history.add(normalize_state(scache.get_state()))
        with torch.no_grad():
            enc_out = enc(torch.FloatTensor(history.get_sequence()).unsqueeze(0))
        action = agent.choose_action(enc_out.detach().numpy(), evaluate=True)
        for q,s in enumerate(agent.action_map[action]):
            if s: scache.scale_queue(q,s)

csr = cold/total*100
print(f'Seed 5 CSR (100k calls): {csr:.3f}%')
print(f'vs Default seed 42: 72.111%')
print(f'Difference: {csr-72.111:+.3f} pp')