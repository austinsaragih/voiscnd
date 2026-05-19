# -*- coding: utf-8 -*-
"""
VOI Evaluator - 10x10 Grid - Single Run 1 of 3 (seed_offset=0)
Methods A-G only (Exact Monolithic excluded)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import timeit
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.setParam('LogToConsole', 0)
env.setParam('MIPGap', 0.01)
env.start()

ACTIVE_SETUP = "10x10"

SETUPS = {
    "3x2":   {"x": 3,  "y": 2,  "cap": 600},
    "3x3":   {"x": 3,  "y": 3,  "cap": 800},
    "5x2":   {"x": 5,  "y": 2,  "cap": 550},
    "4x3":   {"x": 4,  "y": 3,  "cap": 700},
    "5x3":   {"x": 5,  "y": 3,  "cap": 600},
    "5x5":   {"x": 5,  "y": 5,  "cap": 700},
    "7x7":   {"x": 7,  "y": 7,  "cap": 1300},
    "10x10": {"x": 10, "y": 10, "cap": 2500},
    "15x15": {"x": 15, "y": 15, "cap": 6500}
}

COST_FACTOR = 5
FIXED_COST = 25000
RECOURSE_COST = 500
INFO_COST_PER_NODE = 25

N_SAA_BATCHES = 3    
SCENARIOS_PER_BATCH = 10 
BASE_CV = 0.5        
ZONE_CV = 0.25       

# =============================================================================
# 2. DATA GENERATION MODULE
# =============================================================================
def euclidean(x1, y1, x2, y2):
    return np.linalg.norm(np.array((x1, y1)) - np.array((x2, y2)))

def generate_grid_data(setup_key):
    x_blocks = SETUPS[setup_key]["x"]
    y_blocks = SETUPS[setup_key]["y"]
    cases = x_blocks * y_blocks

    np.random.seed(10)
    mu, sigma = 100, 100 * BASE_CV
    s = abs(np.random.normal(mu, sigma, max(200, cases * 2)))
    demand_areas = s[:cases]

    demand_coord_x = np.arange(0.5, x_blocks, 1).tolist() * y_blocks
    demand_coord_y = [val for val in np.arange(0.5, y_blocks, 1)[::-1] for _ in range(x_blocks)]
    dc_coord_x = np.arange(1, x_blocks, 1).tolist() * (y_blocks - 1)
    dc_coord_y = [val for val in np.arange(1, y_blocks, 1)[::-1] for _ in range(x_blocks - 1)]
    if not dc_coord_x: dc_coord_x = [1.0]
    if not dc_coord_y: dc_coord_y = [1.0]

    Demand_Scenarios, Demand_Probs, Demand_Avgs = [], [], []
    for j, base_dem in enumerate(demand_areas):
        np.random.seed(10) 
        demand_j = np.random.normal(base_dem, base_dem * ZONE_CV, 1000)
        demand_j[demand_j < 0] = 0
        n, bins = np.histogram(demand_j, 3)
        probs = n / sum(n)
        scens = np.array([(bins[i] + bins[i+1]) / 2.0 for i in range(len(bins)-1)])
        Demand_Scenarios.append(scens)
        Demand_Probs.append(probs)
        Demand_Avgs.append(probs @ scens)

    # =========================================================================
    # OVERWRITE DC COORDINATES FOR SPECIFIC SETUPS
    # =========================================================================
    if setup_key == "10x10":
        dc_coord_x = [2.5, 5.0, 7.5] * 3
        dc_coord_y = [2.5]*3 + [5.0]*3 + [7.5]*3
    elif setup_key == "15x15":
        dc_coord_x = [3.5, 11.5, 7.5, 3.5, 11.5]
        dc_coord_y = [11.5, 11.5, 7.5, 3.5, 3.5]
    elif setup_key == "7x7":
        dc_coord_x = [1.5, 1.5, 3.5, 5.5, 5.5]
        dc_coord_y = [1.5, 5.5, 3.5, 1.5, 5.5]
    elif setup_key == "5x5":
        dc_coord_x = [1, 1, 2.5, 4, 4]
        dc_coord_y = [1, 4, 2.5, 1, 4]
    elif setup_key == "5x3":
        dc_coord_x = [2.5, 1.5, 1.5, 3.5, 3.5]
        dc_coord_y = [1.5, 0.5, 2.5, 0.5, 2.5]

    return demand_areas, demand_coord_x, demand_coord_y, dc_coord_x, dc_coord_y, Demand_Scenarios, Demand_Probs, Demand_Avgs

print(f"\n[INIT] Generating Data for {ACTIVE_SETUP} Grid...")
capacity = SETUPS[ACTIVE_SETUP]["cap"]
(demand_areas, x_demand, y_demand, x_dc, y_dc, D_Scenarios, D_Probs, D_Avgs) = generate_grid_data(ACTIVE_SETUP)

G_nodes = list(range(len(demand_areas)))
I_nodes = list(range(len(x_dc)))

def CGI(phi): return len(phi) * INFO_COST_PER_NODE
def calc_variance(z): return np.dot((D_Scenarios[z] - D_Avgs[z])**2, D_Probs[z])

# =============================================================================
# 3. OPTIMIZATION MODELS 
# =============================================================================
def Evaluate_WS_Phi(phi, seed):
    J_Stoch = [j for j in G_nodes if j not in phi]
    IGS = list(phi)
    
    np.random.seed(seed)
    Scen_W = np.array([np.random.choice(D_Scenarios[i], SCENARIOS_PER_BATCH, p=D_Probs[i]) for i in J_Stoch]).T if J_Stoch else np.zeros((SCENARIOS_PER_BATCH, 0))
    Scen_Phi = np.array([np.random.choice(D_Scenarios[i], SCENARIOS_PER_BATCH, p=D_Probs[i]) for i in IGS]).T if IGS else np.zeros((SCENARIOS_PER_BATCH, 0))
        
    prob_w = prob_phi = 1.0 / SCENARIOS_PER_BATCH
    q_i_j = {(i, j): euclidean(x_dc[i], y_dc[i], x_demand[j], y_demand[j]) * COST_FACTOR for i in I_nodes for j in G_nodes}
    
    m = gp.Model('WS_Phi', env=env)
    x = m.addVars(I_nodes, range(SCENARIOS_PER_BATCH), vtype=GRB.BINARY)
    y = m.addVars(I_nodes, G_nodes, range(SCENARIOS_PER_BATCH), range(SCENARIOS_PER_BATCH), vtype=GRB.CONTINUOUS)
    h = m.addVars(G_nodes, range(SCENARIOS_PER_BATCH), range(SCENARIOS_PER_BATCH), vtype=GRB.CONTINUOUS)
    
    obj = gp.quicksum(FIXED_COST * x[i, f] * prob_phi for i in I_nodes for f in range(SCENARIOS_PER_BATCH)) + \
          gp.quicksum((q_i_j[i, j] * y[i, j, w, f] + RECOURSE_COST * h[j, w, f]) * prob_w * prob_phi 
                      for i in I_nodes for j in G_nodes for w in range(SCENARIOS_PER_BATCH) for f in range(SCENARIOS_PER_BATCH))
    m.setObjective(obj, GRB.MINIMIZE)
    
    for f in range(SCENARIOS_PER_BATCH):
        for w in range(SCENARIOS_PER_BATCH):
            for i in I_nodes: m.addConstr(gp.quicksum(y[i, j, w, f] for j in G_nodes) <= capacity * x[i, f])
            for idx, j in enumerate(J_Stoch): m.addConstr(gp.quicksum(y[i, j, w, f] for i in I_nodes) + h[j, w, f] >= Scen_W[w][idx])
            for idx, j in enumerate(IGS): m.addConstr(gp.quicksum(y[i, j, w, f] for i in I_nodes) + h[j, w, f] >= Scen_Phi[f][idx])
            
    m.optimize()
    val = m.ObjVal
    m.dispose() 
    return val

def Solve_EV_Design():
    m = gp.Model('EV', env=env)
    x = m.addVars(I_nodes, vtype=GRB.BINARY)
    y = m.addVars(I_nodes, G_nodes, vtype=GRB.CONTINUOUS)
    h = m.addVars(G_nodes, vtype=GRB.CONTINUOUS)
    q_i_j = {(i,j): euclidean(x_dc[i],y_dc[i],x_demand[j],y_demand[j])*COST_FACTOR for i in I_nodes for j in G_nodes}
    m.setObjective(gp.quicksum(FIXED_COST * x[i] for i in I_nodes) + 
                   gp.quicksum(q_i_j[i,j] * y[i,j] for i in I_nodes for j in G_nodes) + 
                   gp.quicksum(RECOURSE_COST * h[j] for j in G_nodes), GRB.MINIMIZE)
    for j in G_nodes: m.addConstr(gp.quicksum(y[i,j] for i in I_nodes) + h[j] >= D_Avgs[j])
    for i in I_nodes: m.addConstr(gp.quicksum(y[i,j] for j in G_nodes) <= capacity * x[i])
    
    m.optimize()
    res = {i: x[i].X for i in I_nodes}
    m.dispose() 
    return res

def Solve_Robust_Design(seed):
    np.random.seed(seed)
    Scen_W = np.array([np.random.choice(D_Scenarios[i], SCENARIOS_PER_BATCH, p=D_Probs[i]) for i in G_nodes]).T
    q_i_j = {(i,j): euclidean(x_dc[i],y_dc[i],x_demand[j],y_demand[j])*COST_FACTOR for i in I_nodes for j in G_nodes}
    m = gp.Model('Robust', env=env)
    x = m.addVars(I_nodes, vtype=GRB.BINARY)
    y = m.addVars(I_nodes, G_nodes, range(SCENARIOS_PER_BATCH), vtype=GRB.CONTINUOUS)
    h = m.addVars(G_nodes, range(SCENARIOS_PER_BATCH), vtype=GRB.CONTINUOUS)
    theta = m.addVar(vtype=GRB.CONTINUOUS)
    m.setObjective(theta, GRB.MINIMIZE)
    first_stage = gp.quicksum(FIXED_COST * x[i] for i in I_nodes)
    for w in range(SCENARIOS_PER_BATCH):
        sec_stage = gp.quicksum(q_i_j[i,j] * y[i,j,w] for i in I_nodes for j in G_nodes) + gp.quicksum(RECOURSE_COST * h[j,w] for j in G_nodes)
        m.addConstr(theta >= first_stage + sec_stage)
        for idx, j in enumerate(G_nodes): m.addConstr(gp.quicksum(y[i,j,w] for i in I_nodes) + h[j,w] >= Scen_W[w][idx])
        for i in I_nodes: m.addConstr(gp.quicksum(y[i,j,w] for j in G_nodes) <= capacity * x[i])
        
    m.optimize()
    res = {i: x[i].X for i in I_nodes}
    m.dispose() 
    return res

def Evaluate_Fixed_Design(fixed_x, seed):
    np.random.seed(seed)
    Scen_W = np.array([np.random.choice(D_Scenarios[i], SCENARIOS_PER_BATCH, p=D_Probs[i]) for i in G_nodes]).T
    prob_w = 1.0 / SCENARIOS_PER_BATCH
    q_i_j = {(i,j): euclidean(x_dc[i],y_dc[i],x_demand[j],y_demand[j])*COST_FACTOR for i in I_nodes for j in G_nodes}
    m = gp.Model('Eval', env=env)
    y = m.addVars(I_nodes, G_nodes, range(SCENARIOS_PER_BATCH), vtype=GRB.CONTINUOUS)
    h = m.addVars(G_nodes, range(SCENARIOS_PER_BATCH), vtype=GRB.CONTINUOUS)
    first_stage = sum(FIXED_COST * fixed_x[i] for i in I_nodes)
    m.setObjective(gp.quicksum((q_i_j[i,j] * y[i,j,w] + RECOURSE_COST * h[j,w]) * prob_w 
                               for i in I_nodes for j in G_nodes for w in range(SCENARIOS_PER_BATCH)), GRB.MINIMIZE)
    for w in range(SCENARIOS_PER_BATCH):
        for idx, j in enumerate(G_nodes): m.addConstr(gp.quicksum(y[i,j,w] for i in I_nodes) + h[j,w] >= Scen_W[w][idx])
        for i in I_nodes: m.addConstr(gp.quicksum(y[i,j,w] for j in G_nodes) <= capacity * fixed_x[i])
        
    m.optimize()
    val = first_stage + m.ObjVal
    m.dispose() 
    return val

# =============================================================================
# 4. SAA WRAPPERS 
# =============================================================================
def SAA_ETSCI(phi, offset_seed=0):
    vals = [Evaluate_WS_Phi(phi, offset_seed + seed) for seed in range(N_SAA_BATCHES)]
    return np.mean(vals) + CGI(phi)

def SAA_Fixed_Eval(fixed_x, offset_seed=0):
    vals = [Evaluate_Fixed_Design(fixed_x, offset_seed + seed) for seed in range(N_SAA_BATCHES)]
    return np.mean(vals)

# =============================================================================
# 5. SINGLE RUN (Run 3 of 3, seed_offset=20)
# =============================================================================
seed_offset = 20  # Run 3
print(f"\n==================== RUN 3/3 (seed_offset={seed_offset}) ====================")

results = {}

print(f"[Run 3] Running RP Evaluation...")
st = timeit.default_timer()
RP_val = SAA_ETSCI([], seed_offset)
results['No Information (RP)'] = {'val': RP_val, 'time': timeit.default_timer() - st}

print(f"[Run 3] Running WS Evaluation...")
st = timeit.default_timer()
WS_val = SAA_ETSCI(G_nodes, seed_offset) - CGI(G_nodes)
results['Full Information (WS)'] = {'val': WS_val + CGI(G_nodes), 'time': timeit.default_timer() - st}

print(f"[Run 3] Running Deterministic EV Evaluation...")
st = timeit.default_timer()
x_ev = Solve_EV_Design()
results['Deterministic EV'] = {'val': SAA_Fixed_Eval(x_ev, seed_offset), 'time': timeit.default_timer() - st}

print(f"[Run 3] Running Robust Minimax Evaluation...")
st = timeit.default_timer()
x_rob = Solve_Robust_Design(seed_offset)
results['Robust Minimax'] = {'val': SAA_Fixed_Eval(x_rob, seed_offset), 'time': timeit.default_timer() - st}

# --- C. Greedy EVPPI/Cost ---
print(f"[Run 3] Running Greedy EVPPI/Cost...")
st = timeit.default_timer()
EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}
best_greedy_val = RP_val
curr_phi = []
while len(curr_phi) < len(G_nodes):
    best_step_val = float('inf')
    best_node = None
    for i in G_nodes:
        if i not in curr_phi:
            candidate_phi = curr_phi + [i]
            val = SAA_ETSCI(candidate_phi, seed_offset)
            if val < best_step_val:
                best_step_val = val
                best_node = i
    if best_step_val < best_greedy_val:
        best_greedy_val = best_step_val
        curr_phi.append(best_node)
    else:
        break
results['Greedy EVPPI/Cost'] = {'val': best_greedy_val, 'time': timeit.default_timer() - st, 'size': len(curr_phi)}

# --- D. Greedy Variance/Cost ---
print(f"[Run 3] Running Greedy Variance/Cost...")
st = timeit.default_timer()
ratios_var = {i: calc_variance(i) / (CGI([i]) + 1e-9) for i in G_nodes}
sorted_var = sorted(G_nodes, key=lambda i: ratios_var[i], reverse=True)
best_greedy_var_val = RP_val
curr_phi_var = []
for node in sorted_var:
    candidate_phi = curr_phi_var + [node]
    val = SAA_ETSCI(candidate_phi, seed_offset)
    if val < best_greedy_var_val:
        best_greedy_var_val = val
        curr_phi_var.append(node)
results['Greedy Variance/Cost'] = {'val': best_greedy_var_val, 'time': timeit.default_timer() - st, 'size': len(curr_phi_var)}

# --- E. Top-5 EVPPI ---
print(f"[Run 3] Running Top-5 EVPPI...")
st = timeit.default_timer()
EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}
sorted_top_evppi = sorted(G_nodes, key=lambda i: EVPPI_i[i], reverse=True)
top_5_phi = sorted_top_evppi[:5]
val_top_5 = SAA_ETSCI(top_5_phi, seed_offset)
results['Top-5 EVPPI'] = {'val': val_top_5, 'time': timeit.default_timer() - st, 'size': len(top_5_phi)}

# --- F. Bounds (Proposed) ---
print(f"[Run 3] Running Bounds Approximation...")
st = timeit.default_timer()
EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}
phi_bounds = [i for i in G_nodes if CGI([i]) - EVPPI_i[i] <= 0]
val_bounds = SAA_ETSCI(phi_bounds, seed_offset)
if val_bounds > RP_val: val_bounds, phi_bounds = RP_val, []
if val_bounds > (WS_val + CGI(G_nodes)): val_bounds, phi_bounds = WS_val + CGI(G_nodes), G_nodes
results['Bounds (Proposed)'] = {'val': val_bounds, 'time': timeit.default_timer() - st, 'size': len(phi_bounds)}

# =============================================================================
# 6. REPORTING
# =============================================================================
print("\n" + "="*80)
print(f"RESULTS | Run 3/3 (seed_offset={seed_offset}) | Setup: {ACTIVE_SETUP} | Nodes: {len(G_nodes)}")
print(f"{'Method':<25} | {'ETSCI ($)':<20} | {'Time (s)':<15} | {'IGS Size'}")
print("-" * 80)
for name in ['No Information (RP)', 'Full Information (WS)']:
    r = results[name]
    print(f"{name:<25} | {r['val']:<20.2f} | {r['time']:<15.4f} | Fixed")
print("-" * 80)
for name in ['Deterministic EV', 'Robust Minimax', 'Greedy EVPPI/Cost', 'Greedy Variance/Cost', 'Top-5 EVPPI', 'Bounds (Proposed)']:
    r = results[name]
    sz = "Fixed" if name in ['Deterministic EV', 'Robust Minimax'] else str(r.get('size', ''))
    print(f"{name:<25} | {r['val']:<20.2f} | {r['time']:<15.4f} | {sz}")
print("=" * 80)
