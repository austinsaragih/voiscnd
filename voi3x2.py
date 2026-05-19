# -*- coding: utf-8 -*-
"""
Supply Chain Network Design (SCND) Value of Information (VOI) Evaluator
- Includes Macro-Runs for Mean/Std Deviation
- Includes Top-5 EVPPI Heuristic
- Includes all setups up to 15x15
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import timeit
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 0. SILENCE GUROBI COMPLETELY
# =============================================================================
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.setParam('LogToConsole', 0)
env.setParam('MIPGap', 0.01)
env.start()

# =============================================================================
# 1. GLOBAL CONFIGURATION & TEST CASES
# =============================================================================
# ---> UNCOMMENT THE TARGET SETUP HERE <---
# ACTIVE_SETUP = "3x3"
ACTIVE_SETUP = "3x2"
# ACTIVE_SETUP = "5x2"
# ACTIVE_SETUP = "4x3"
# ACTIVE_SETUP = "5x3"
# ACTIVE_SETUP = "5x5"
# ACTIVE_SETUP = "7x7"
# ACTIVE_SETUP = "10x10"
# ACTIVE_SETUP = "15x15"

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
FIXED_COST = 5000
RECOURSE_COST = 500
INFO_COST_PER_NODE = 200

N_SAA_BATCHES = 3   
SCENARIOS_PER_BATCH = 5 
BASE_CV = 0.5        
ZONE_CV = 0.25       
N_MACRO_RUNS = 3     
EXACT_TIME_LIMIT = 10000 # Seconds limit for Exact Monolithic

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
# 5. MACRO-RUN SIMULATION ENGINE (FOR 95% CIs)
# =============================================================================
metrics = {k: {'val': [], 'time': [], 'size': []} for k in [
    'No Information (RP)', 'Full Information (WS)', 'Deterministic EV', 'Robust Minimax',
    'Greedy EVPPI/Cost', 'Greedy Variance/Cost', 'Top-5 EVPPI', 'Bounds (Proposed)', 'APSM (Proposed)', 'Exact (Monolithic)'
]}

for run in range(N_MACRO_RUNS):
    print(f"\n==================== MACRO RUN {run+1}/{N_MACRO_RUNS} ====================")
    seed_offset = run * 10 
    print(f"[{run+1}/{N_MACRO_RUNS}] Running RP Evaluation...")
    st = timeit.default_timer()
    RP_val = SAA_ETSCI([], seed_offset)
    metrics['No Information (RP)']['val'].append(RP_val)
    metrics['No Information (RP)']['time'].append(timeit.default_timer() - st)
    
    print(f"[{run+1}/{N_MACRO_RUNS}] Running WS Evaluation...")
    st = timeit.default_timer()
    WS_val = SAA_ETSCI(G_nodes, seed_offset) - CGI(G_nodes)
    metrics['Full Information (WS)']['val'].append(WS_val + CGI(G_nodes))
    metrics['Full Information (WS)']['time'].append(timeit.default_timer() - st)
    
    print(f"[{run+1}/{N_MACRO_RUNS}] Running Deterministic EV Evaluation...")
    # --- A. Deterministic EV ---
    st = timeit.default_timer()
    x_ev = Solve_EV_Design()
    metrics['Deterministic EV']['val'].append(SAA_Fixed_Eval(x_ev, seed_offset))
    metrics['Deterministic EV']['time'].append(timeit.default_timer() - st)

    print(f"[{run+1}/{N_MACRO_RUNS}] Running Robust Minimax Evaluation...")
    # --- B. Robust Minimax ---
    st = timeit.default_timer()
    x_rob = Solve_Robust_Design(seed_offset)
    metrics['Robust Minimax']['val'].append(SAA_Fixed_Eval(x_rob, seed_offset))
    metrics['Robust Minimax']['time'].append(timeit.default_timer() - st)

    # --- C. Greedy EVPPI/Cost (Prefix Evaluation) ---
    print(f"[{run+1}/{N_MACRO_RUNS}] Running Greedy EVPPI/Cost...")
    st = timeit.default_timer()
    
    # 1. We still need to calculate EVPPI_i once for the Top-5 and Bounds heuristics to use later
    EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}

    # 2. Start the O(N^2) Greedy Loop
    best_greedy_val = RP_val
    curr_phi = []
    
    while len(curr_phi) < len(G_nodes):
        best_step_val = float('inf')
        best_node = None
        
        # Test adding every single node that hasn't been chosen yet
        for i in G_nodes:
            if i not in curr_phi:
                candidate_phi = curr_phi + [i]
                val = SAA_ETSCI(candidate_phi, seed_offset)
                
                if val < best_step_val:
                    best_step_val = val
                    best_node = i
                    
        # Did adding the best node actually lower our total expected cost?
        if best_step_val < best_greedy_val:
            best_greedy_val = best_step_val
            curr_phi.append(best_node)
            # print(f"      [Greedy Step] Added node {best_node}. New Cost: {best_greedy_val:.2f}")
        else:
            # If cost goes up (because info cost > benefit), stop searching!
            break
            
    metrics['Greedy EVPPI/Cost']['val'].append(best_greedy_val)
    metrics['Greedy EVPPI/Cost']['time'].append(timeit.default_timer() - st)
    metrics['Greedy EVPPI/Cost']['size'].append(len(curr_phi))

    # --- D. Greedy Variance/Cost (Prefix Evaluation) ---
    print(f"[{run+1}/{N_MACRO_RUNS}] Running Greedy Variance/Cost...")
    st = timeit.default_timer()
    
    # 1. Calculate static Variance/Cost ratio for each node
    ratios_var = {i: calc_variance(i) / (CGI([i]) + 1e-9) for i in G_nodes}
    sorted_var = sorted(G_nodes, key=lambda i: ratios_var[i], reverse=True)
    
    # 2. Iterate through the sorted list and conditionally accept
    best_greedy_var_val = RP_val
    curr_phi_var = []
    
    for node in sorted_var:
        candidate_phi = curr_phi_var + [node]
        val = SAA_ETSCI(candidate_phi, seed_offset)
        
        # Only keep the node if the information benefit strictly outweighs the $200 cost
        if val < best_greedy_var_val:
            best_greedy_var_val = val
            curr_phi_var.append(node)
            
    metrics['Greedy Variance/Cost']['val'].append(best_greedy_var_val)
    metrics['Greedy Variance/Cost']['time'].append(timeit.default_timer() - st)
    metrics['Greedy Variance/Cost']['size'].append(len(curr_phi_var))

    # --- E. Top-5 EVPPI ---
    print(f"[{run+1}/{N_MACRO_RUNS}] Running Top-5 EVPPI...")
    st = timeit.default_timer()
    # Evaluate EVPPI for every i (used by Bounds, Top-5, and Greedy EVPPI/Cost)
    EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}
    sorted_top_evppi = sorted(G_nodes, key=lambda i: EVPPI_i[i], reverse=True)
    top_5_phi = sorted_top_evppi[:5] # Takes up to 5 elements safely
    val_top_5 = SAA_ETSCI(top_5_phi, seed_offset)
    metrics['Top-5 EVPPI']['val'].append(val_top_5)
    metrics['Top-5 EVPPI']['time'].append(timeit.default_timer() - st)
    metrics['Top-5 EVPPI']['size'].append(len(top_5_phi))

    # --- F. Bounds (Proposed) ---
    print(f"[{run+1}/{N_MACRO_RUNS}] Running Bounds Approximation...")
    st = timeit.default_timer()
    EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}
    phi_bounds = [i for i in G_nodes if CGI([i]) - EVPPI_i[i] <= 0]
    val_bounds = SAA_ETSCI(phi_bounds, seed_offset)
    if val_bounds > RP_val: val_bounds, phi_bounds = RP_val, []
    if val_bounds > (WS_val + CGI(G_nodes)): val_bounds, phi_bounds = WS_val + CGI(G_nodes), G_nodes
    metrics['Bounds (Proposed)']['val'].append(val_bounds)
    metrics['Bounds (Proposed)']['time'].append(timeit.default_timer() - st)
    metrics['Bounds (Proposed)']['size'].append(len(phi_bounds))

    # --- G. APSM (Proposed) ---
    print(f"[{run+1}/{N_MACRO_RUNS}] Running APSM Algorithm...")
    st = timeit.default_timer()
    def tilde_ETSCI(phi): return SAA_ETSCI(list(set(G_nodes) - set(phi)), seed_offset) - (WS_val + CGI(G_nodes))

    T, N = round(len(G_nodes)/2), len(G_nodes)
    eta = (2 * np.sqrt(N)) / (max(np.abs(sum(EVPPI_i.values()) - (RP_val - WS_val)), 1e-5) * np.sqrt(T))
    varphi = np.full(N, 0.5)

    for t in range(T):
        phi_N = list(np.flip(np.argsort(varphi)))
        kappa = np.zeros(N)
        for l in range(N):
            kappa[phi_N[l]] = tilde_ETSCI(phi_N[:l+1]) - tilde_ETSCI(phi_N[:l])
        varphi = np.clip(varphi - eta * kappa, 0, 1)

    hat_phi_N = list(np.flip(np.argsort(varphi)))
    tilde_phi_idx = min(range(N + 1), key=lambda l: tilde_ETSCI(set(hat_phi_N[:l])))
    S_apsm = sorted(list(set(G_nodes) - set(hat_phi_N[:tilde_phi_idx])))
    metrics['APSM (Proposed)']['val'].append(SAA_ETSCI(S_apsm, seed_offset))
    metrics['APSM (Proposed)']['time'].append(timeit.default_timer() - st)
    metrics['APSM (Proposed)']['size'].append(len(S_apsm))

    # --- H. Exact Monolithic (With 50k Timeout) ---
    print(f"[{run+1}/{N_MACRO_RUNS}] Running Exact Monolithic Evaluation...")
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
    metrics['Exact (Monolithic)']['val'].append(best_exact_val)
    metrics['Exact (Monolithic)']['time'].append(timeit.default_timer() - st)
    metrics['Exact (Monolithic)']['size'].append(len(best_exact_phi))


# =============================================================================
# 6. FINAL REPORTING (95% CIs)
# =============================================================================
print("\n" + "="*115)
print(f"FINAL AGGREGATED RESULTS ({N_MACRO_RUNS} Macro-Runs) | Setup: {ACTIVE_SETUP} | Nodes: {len(G_nodes)}")
print(f"{'Method':<25} | {'95% CI ETSCI ($)':<30} | {'95% CI Time (s)':<30} | {'Mean IGS Size'}")
print("-" * 115)

T_CRIT = 4.303  # t_0.025 for df = N_MACRO_RUNS - 1 = 2
def fmt_ci(arr, is_time=False):
    mean = np.mean(arr)
    ci = T_CRIT * np.std(arr, ddof=1) / np.sqrt(len(arr))
    return f"{mean:.4f} ± {ci:.4f}" if is_time else f"{mean:.2f} ± {ci:.2f}"

def print_row(name, is_fixed=False):
    if name in metrics and len(metrics[name]['val']) > 0:
        sz = "Fixed" if is_fixed else f"{np.mean(metrics[name]['size']):.1f}" if metrics[name]['size'] else "0"
        print(f"{name:<25} | {fmt_ci(metrics[name]['val']):<30} | {fmt_ci(metrics[name]['time'], True):<30} | {sz}")

print_row('No Information (RP)', True)
print_row('Full Information (WS)', True)
print("-" * 115)
for key in ['APSM (Proposed)', 'Deterministic EV', 'Robust Minimax', 'Greedy EVPPI/Cost', 'Greedy Variance/Cost', 'Top-5 EVPPI', 'Bounds (Proposed)', 'Exact (Monolithic)']:
    print_row(key, is_fixed=(key in ['Deterministic EV', 'Robust Minimax']))
print("="*115)
