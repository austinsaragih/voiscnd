# -*- coding: utf-8 -*-
"""
VOI Evaluator - 7x7 Grid - Single Run 3 of 3 (seed_offset=20)
APSM (G) ONLY - Methods A-F, H excluded
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import timeit
import warnings

warnings.filterwarnings("ignore")

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.setParam('LogToConsole', 0)
env.setParam('MIPGap', 0.01)
env.start()

ACTIVE_SETUP = "7x7"

SETUPS = {
    "3x2":   {"x": 3,  "y": 2,  "cap": 600},
    "3x3":   {"x": 3,  "y": 3,  "cap": 800},
    "5x2":   {"x": 5,  "y": 2,  "cap": 550},
    "4x3":   {"x": 4,  "y": 3,  "cap": 700},
    "5x3":   {"x": 5,  "y": 3,  "cap": 800},
    "5x5":   {"x": 5,  "y": 5,  "cap": 900},
    "7x7":   {"x": 7,  "y": 7,  "cap": 1300},
    "10x10": {"x": 10, "y": 10, "cap": 2000},
    "15x15": {"x": 15, "y": 15, "cap": 6500}
}

COST_FACTOR = 5
FIXED_COST = 15000
RECOURSE_COST = 500
INFO_COST_PER_NODE = 50

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
        dc_coord_x = [2, 8, 5, 2, 8]
        dc_coord_y = [8, 8, 5, 2, 2]
    elif setup_key == "15x15":
        dc_coord_x = [3.5, 11.5, 7.5, 3.5, 11.5]
        dc_coord_y = [11.5, 11.5, 7.5, 3.5, 3.5]
    elif setup_key == "7x7":
        dc_coord_x = [2, 4, 6] * 3
        dc_coord_y = [2]*3 + [4]*3 + [6]*3
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
# 5. SINGLE RUN - APSM ONLY (Run 3 of 3, seed_offset=20)
# =============================================================================
seed_offset = 20  # Run 3
print(f"\n==================== APSM RUN 3/3 (seed_offset={seed_offset}) ====================")

# --- Prerequisites: RP, WS, EVPPI ---
print(f"[APSM Run 3] Computing RP and WS values...")
RP_val = SAA_ETSCI([], seed_offset)
WS_val = SAA_ETSCI(G_nodes, seed_offset) - CGI(G_nodes)
print(f"[APSM Run 3] Computing EVPPI values...")
EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}

# --- G. APSM (Proposed) ---
print(f"[APSM Run 3] Running APSM Algorithm...")
st = timeit.default_timer()
def tilde_ETSCI(phi): return SAA_ETSCI(list(set(G_nodes) - set(phi)), seed_offset) - (WS_val + CGI(G_nodes))

APSM_TIME_LIMIT = 30000
T, N = 2, len(G_nodes)
eta = (2 * np.sqrt(N)) / (max(np.abs(sum(EVPPI_i.values()) - (RP_val - WS_val)), 1e-5) * np.sqrt(T))
varphi = np.full(N, 0.5)

timeout = False
for t in range(T):
    if timeit.default_timer() - st > APSM_TIME_LIMIT:
        print(f"   --> Hit {APSM_TIME_LIMIT}s Time Limit at iteration {t}/{T}! Breaking.")
        timeout = True
        break
    phi_N = list(np.flip(np.argsort(varphi)))
    kappa = np.zeros(N)
    for l in range(N):
        kappa[phi_N[l]] = tilde_ETSCI(phi_N[:l+1]) - tilde_ETSCI(phi_N[:l])
    varphi = np.clip(varphi - eta * kappa, 0, 1)

hat_phi_N = list(np.flip(np.argsort(varphi)))
tilde_phi_idx = min(range(N + 1), key=lambda l: tilde_ETSCI(set(hat_phi_N[:l])))
S_apsm = sorted(list(set(G_nodes) - set(hat_phi_N[:tilde_phi_idx])))
apsm_val = SAA_ETSCI(S_apsm, seed_offset)
apsm_time = timeit.default_timer() - st

# =============================================================================
# 6. REPORTING
# =============================================================================
print("\n" + "="*80)
print(f"RESULTS | APSM Run 3/3 (seed_offset={seed_offset}) | Setup: {ACTIVE_SETUP} | Nodes: {len(G_nodes)}")
print(f"{'Method':<25} | {'ETSCI ($)':<20} | {'Time (s)':<15} | {'IGS Size'}")
print("-" * 80)
print(f"{'APSM (Proposed)':<25} | {apsm_val:<20.2f} | {apsm_time:<15.4f} | {len(S_apsm)}")
print(f"Best Phi: {S_apsm}")
print(f"Timed out: {timeout}")
print("=" * 80)
