# -*- coding: utf-8 -*-
"""
VOI Evaluator - 15x15 Grid - Single Run 1 of 3 (seed_offset=0)
Exact Monolithic (H) ONLY - Methods A-G excluded
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

ACTIVE_SETUP = "15x15"

SETUPS = {
    "3x2":   {"x": 3,  "y": 2,  "cap": 600},
    "3x3":   {"x": 3,  "y": 3,  "cap": 800},
    "5x2":   {"x": 5,  "y": 2,  "cap": 550},
    "4x3":   {"x": 4,  "y": 3,  "cap": 700},
    "5x3":   {"x": 5,  "y": 3,  "cap": 600},
    "5x5":   {"x": 5,  "y": 5,  "cap": 700},
    "7x7":   {"x": 7,  "y": 7,  "cap": 1300},
    "10x10": {"x": 10, "y": 10, "cap": 2500},
    "15x15": {"x": 15, "y": 15, "cap": 6000}
}

COST_FACTOR = 5
FIXED_COST = 50000
RECOURSE_COST = 500
INFO_COST_PER_NODE = 25

N_SAA_BATCHES = 3    
SCENARIOS_PER_BATCH = 10 
BASE_CV = 0.5        
ZONE_CV = 0.25         
EXACT_TIME_LIMIT = 30000 # Seconds limit for Exact Monolithic

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
        dc_coord_x = [2, 8, 5, 2, 8]
        dc_coord_y = [8, 8, 5, 2, 2]
    elif setup_key == "15x15":
        dc_coord_x = [3.5, 7.5, 11.5] * 3
        dc_coord_y = [3.5]*3 + [7.5]*3 + [11.5]*3
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

# =============================================================================
# 4. SAA WRAPPERS 
# =============================================================================
def SAA_ETSCI(phi, offset_seed=0):
    vals = [Evaluate_WS_Phi(phi, offset_seed + seed) for seed in range(N_SAA_BATCHES)]
    return np.mean(vals) + CGI(phi)

# =============================================================================
# 5. SINGLE RUN - EXACT MONOLITHIC ONLY (Run 1 of 3, seed_offset=0)
# =============================================================================
seed_offset = 0  # Run 1
print(f"\n==================== EXACT RUN 1/3 (seed_offset={seed_offset}) ====================")

# --- H. Exact Monolithic ---
print(f"[Exact Run 1] Running Exact Monolithic Evaluation...")
st = timeit.default_timer()
best_exact_val, best_exact_phi, timeout = float('inf'), [], False
for r in range(len(G_nodes) + 1):
    if timeout: break
    for subset in combinations(G_nodes, r):
        if timeit.default_timer() - st > EXACT_TIME_LIMIT:
            print(f"   --> Hit {EXACT_TIME_LIMIT}s Time Limit! Breaking.")
            timeout = True
            break
        val = SAA_ETSCI(list(subset), seed_offset)
        if val < best_exact_val:
            best_exact_val, best_exact_phi = val, list(subset)
exact_time = timeit.default_timer() - st

# =============================================================================
# 6. REPORTING
# =============================================================================
print("\n" + "="*80)
print(f"RESULTS | Exact Run 1/3 (seed_offset={seed_offset}) | Setup: {ACTIVE_SETUP} | Nodes: {len(G_nodes)}")
print(f"{'Method':<25} | {'ETSCI ($)':<20} | {'Time (s)':<15} | {'IGS Size'}")
print("-" * 80)
print(f"{'Exact (Monolithic)':<25} | {best_exact_val:<20.2f} | {exact_time:<15.4f} | {len(best_exact_phi)}")
print(f"Best Phi: {best_exact_phi}")
print(f"Timed out: {timeout}")
print("=" * 80)
