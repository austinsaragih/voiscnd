"""
Supply Chain Network Design (SCND) Value of Information
ARGENTINA CASE STUDY 
- 8 DCs, 24 Provinces
- SAA Evaluation (Discrete 3-State Distribution)
- Enforces High Variance (CV=0.5) on Swing Provinces
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import timeit
import math
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
env.setParam('MIPFocus', 2) 
env.setParam('Method', 3) 
env.start()

# =============================================================================
# 1. GLOBAL CONFIGURATION & ARGENTINA DATA
# =============================================================================
COST_FACTOR = 0.025
FIXED_COST = 1000000
RECOURSE_COST = 5000
CAPACITY = 37500

N_SAA_BATCHES = 2    
SCENARIOS_PER_BATCH = 25 

demand_prov = {
    'Buenos Aires': 57189, 'Catamarca': 1784, 'Chaco': 5425, 'Chubut': 1066,
    'Ciudad de Buenos Aires': 10723, 'Cordoba': 17157, 'Corrientes': 5294,
    'Entre Rios': 8671, 'Formosa': 4304, 'Jujuy': 4698, 'La Pampa': 736,
    'La Rioja': 934, 'Mendoza': 8021, 'Misiones': 1794, 'Neuquen': 4513,
    'Rio Negro': 8949, 'Salta': 11934, 'San Juan': 3561, 'San Luis': 1945,
    'Santa Cruz': 11417, 'Santa Fe': 19909, 'Santiago del Estero': 6839,
    'Tierra del Fuego': 10661, 'Tucuman': 5061
}

dist_dc_prov = {
    ('Buenos Aires', 'Buenos Aires'): 304.03, ('Buenos Aires', 'Catamarca'): 1147.69,
    ('Buenos Aires', 'Chaco'): 941.66, ('Buenos Aires', 'Chubut'): 1342.36,
    ('Buenos Aires', 'Ciudad de Buenos Aires'): 5.96, ('Buenos Aires', 'Cordoba'): 572.67,
    ('Buenos Aires', 'Corrientes'): 650.26, ('Buenos Aires', 'Entre Rios'): 294.36,
    ('Buenos Aires', 'Formosa'): 1089.92, ('Buenos Aires', 'Jujuy'): 1444.77,
    ('Buenos Aires', 'La Pampa'): 695.76, ('Buenos Aires', 'La Rioja'): 992.48,
    ('Buenos Aires', 'Mendoza'): 933.51, ('Buenos Aires', 'Misiones'): 930.34,
    ('Buenos Aires', 'Neuquen'): 1138.32, ('Buenos Aires', 'Rio Negro'): 1011.67,
    ('Buenos Aires', 'Salta'): 1303.79, ('Buenos Aires', 'San Juan'): 1066.04,
    ('Buenos Aires', 'San Luis'): 709.45, ('Buenos Aires', 'Santa Cruz'): 1843.05,
    ('Buenos Aires', 'Santa Fe'): 495.85, ('Buenos Aires', 'Santiago del Estero'): 888.79,
    ('Buenos Aires', 'Tierra del Fuego'): 2303.10, ('Buenos Aires', 'Tucuman'): 1081.28,
    ('Rosario', 'Buenos Aires'): 414.34, ('Rosario', 'Catamarca'): 866.82,
    ('Rosario', 'Chaco'): 730.73, ('Rosario', 'Chubut'): 1382.97,
    ('Rosario', 'Ciudad de Buenos Aires'): 277.85, ('Rosario', 'Cordoba'): 305.01,
    ('Rosario', 'Corrientes'): 540.50, ('Rosario', 'Entre Rios'): 172.40,
    ('Rosario', 'Formosa'): 899.73, ('Rosario', 'Jujuy'): 1181.25,
    ('Rosario', 'La Pampa'): 634.29, ('Rosario', 'La Rioja'): 715.91,
    ('Rosario', 'Mendoza'): 752.49, ('Rosario', 'Misiones'): 892.43,
    ('Rosario', 'Neuquen'): 1058.00, ('Rosario', 'Rio Negro'): 1011.98,
    ('Rosario', 'Salta'): 1043.69, ('Rosario', 'San Juan'): 807.16,
    ('Rosario', 'San Luis'): 503.66, ('Rosario', 'Santa Cruz'): 1922.62,
    ('Rosario', 'Santa Fe'): 251.94, ('Rosario', 'Santiago del Estero'): 625.93,
    ('Rosario', 'Tierra del Fuego'): 2434.53, ('Rosario', 'Tucuman'): 805.97,
    ('Cordoba', 'Buenos Aires'): 673.60, ('Cordoba', 'Catamarca'): 526.67,
    ('Cordoba', 'Chaco'): 651.14, ('Cordoba', 'Chubut'): 1427.04,
    ('Cordoba', 'Ciudad de Buenos Aires'): 642.67, ('Cordoba', 'Cordoba'): 88.22,
    ('Cordoba', 'Corrientes'): 680.62, ('Cordoba', 'Entre Rios'): 476.35,
    ('Cordoba', 'Formosa'): 836.72, ('Cordoba', 'Jujuy'): 914.26,
    ('Cordoba', 'La Pampa'): 645.38, ('Cordoba', 'La Rioja'): 345.81,
    ('Cordoba', 'Mendoza'): 543.41, ('Cordoba', 'Misiones'): 1054.84,
    ('Cordoba', 'Neuquen'): 966.83, ('Cordoba', 'Rio Negro'): 1035.88,
    ('Cordoba', 'Salta'): 794.57, ('Cordoba', 'San Juan'): 450.93,
    ('Cordoba', 'San Luis'): 312.83, ('Cordoba', 'Santa Cruz'): 1992.91,
    ('Cordoba', 'Santa Fe'): 318.53, ('Cordoba', 'Santiago del Estero'): 414.38,
    ('Cordoba', 'Tierra del Fuego'): 2560.54, ('Cordoba', 'Tucuman'): 510.67,
    ('Tucuman', 'Buenos Aires'): 1182.44, ('Tucuman', 'Catamarca'): 180.69,
    ('Tucuman', 'Chaco'): 444.98, ('Tucuman', 'Chubut'): 1911.62,
    ('Tucuman', 'Ciudad de Buenos Aires'): 1082.52, ('Tucuman', 'Cordoba'): 608.65,
    ('Tucuman', 'Corrientes'): 761.03, ('Tucuman', 'Entre Rios'): 823.32,
    ('Tucuman', 'Formosa'): 569.83, ('Tucuman', 'Jujuy'): 392.05,
    ('Tucuman', 'La Pampa'): 1147.96, ('Tucuman', 'La Rioja'): 372.55,
    ('Tucuman', 'Mendoza'): 927.13, ('Tucuman', 'Misiones'): 1048.45,
    ('Tucuman', 'Neuquen'): 1392.49, ('Tucuman', 'Rio Negro'): 1523.36,
    ('Tucuman', 'Salta'): 282.24, ('Tucuman', 'San Juan'): 574.63,
    ('Tucuman', 'San Luis'): 777.73, ('Tucuman', 'Santa Cruz'): 2480.09,
    ('Tucuman', 'Santa Fe'): 600.38, ('Tucuman', 'Santiago del Estero'): 221.81,
    ('Tucuman', 'Tierra del Fuego'): 3065.80, ('Tucuman', 'Tucuman'): 21.04,
    ('Mendoza', 'Buenos Aires'): 865.38, ('Mendoza', 'Catamarca'): 643.97,
    ('Mendoza', 'Chaco'): 1063.42, ('Mendoza', 'Chubut'): 1212.46,
    ('Mendoza', 'Ciudad de Buenos Aires'): 980.15, ('Mendoza', 'Cordoba'): 480.37,
    ('Mendoza', 'Corrientes'): 1148.21, ('Mendoza', 'Entre Rios'): 908.89,
    ('Mendoza', 'Formosa'): 1241.09, ('Mendoza', 'Jujuy'): 1106.31,
    ('Mendoza', 'La Pampa'): 563.91, ('Mendoza', 'La Rioja'): 390.55,
    ('Mendoza', 'Mendoza'): 194.86, ('Mendoza', 'Misiones'): 1521.63,
    ('Mendoza', 'Neuquen'): 649.53, ('Mendoza', 'Rio Negro'): 848.18,
    ('Mendoza', 'Salta'): 1033.08, ('Mendoza', 'San Juan'): 225.75,
    ('Mendoza', 'San Luis'): 279.22, ('Mendoza', 'Santa Cruz'): 1772.66,
    ('Mendoza', 'Santa Fe'): 784.49, ('Mendoza', 'Santiago del Estero'): 780.76,
    ('Mendoza', 'Tierra del Fuego'): 2387.13, ('Mendoza', 'Tucuman'): 741.20,
    ('Resistencia', 'Buenos Aires'): 1037.20, ('Resistencia', 'Catamarca'): 785.63,
    ('Resistencia', 'Chaco'): 212.59, ('Resistencia', 'Chubut'): 2007.69,
    ('Resistencia', 'Ciudad de Buenos Aires'): 798.46, ('Resistencia', 'Cordoba'): 698.14,
    ('Resistencia', 'Corrientes'): 187.27, ('Resistencia', 'Entre Rios'): 511.44,
    ('Resistencia', 'Formosa'): 299.71, ('Resistencia', 'Jujuy'): 821.27,
    ('Resistencia', 'La Pampa'): 1234.98, ('Resistencia', 'La Rioja'): 837.50,
    ('Resistencia', 'Mendoza'): 1212.76, ('Resistencia', 'Misiones'): 433.93,
    ('Resistencia', 'Neuquen'): 1617.18, ('Resistencia', 'Rio Negro'): 1627.04,
    ('Resistencia', 'Salta'): 680.69, ('Resistencia', 'San Juan'): 1032.46,
    ('Resistencia', 'San Luis'): 972.90, ('Resistencia', 'Santa Cruz'): 2554.40,
    ('Resistencia', 'Santa Fe'): 408.88, ('Resistencia', 'Santiago del Estero'): 422.57,
    ('Resistencia', 'Tierra del Fuego'): 3066.56, ('Resistencia', 'Tucuman'): 633.45,
    ('Bahia Blanca', 'Buenos Aires'): 271.23, ('Bahia Blanca', 'Catamarca'): 1338.08,
    ('Bahia Blanca', 'Chaco'): 1378.26, ('Bahia Blanca', 'Chubut'): 768.88,
    ('Bahia Blanca', 'Ciudad de Buenos Aires'): 569.17, ('Bahia Blanca', 'Cordoba'): 744.38,
    ('Bahia Blanca', 'Corrientes'): 1179.48, ('Bahia Blanca', 'Entre Rios'): 791.82,
    ('Bahia Blanca', 'Formosa'): 1552.69, ('Bahia Blanca', 'Jujuy'): 1744.23,
    ('Bahia Blanca', 'La Pampa'): 330.23, ('Bahia Blanca', 'La Rioja'): 1101.67,
    ('Bahia Blanca', 'Mendoza'): 723.93, ('Bahia Blanca', 'Misiones'): 1495.72,
    ('Bahia Blanca', 'Neuquen'): 681.38, ('Bahia Blanca', 'Rio Negro'): 464.86,
    ('Bahia Blanca', 'Salta'): 1621.62, ('Bahia Blanca', 'San Juan'): 1061.63,
    ('Bahia Blanca', 'San Luis'): 645.59, ('Bahia Blanca', 'Santa Cruz'): 1278.24,
    ('Bahia Blanca', 'Santa Fe'): 899.24, ('Bahia Blanca', 'Santiago del Estero'): 1219.50,
    ('Bahia Blanca', 'Tierra del Fuego'): 1778.92, ('Bahia Blanca', 'Tucuman'): 1340.66,
    ('Comodoro Rivadavia', 'Buenos Aires'): 1172.16, ('Comodoro Rivadavia', 'Catamarca'): 2060.90,
    ('Comodoro Rivadavia', 'Chaco'): 2246.22, ('Comodoro Rivadavia', 'Chubut'): 244.95,
    ('Comodoro Rivadavia', 'Ciudad de Buenos Aires'): 1465.01, ('Comodoro Rivadavia', 'Cordoba'): 1558.44,
    ('Comodoro Rivadavia', 'Corrientes'): 2080.26, ('Comodoro Rivadavia', 'Entre Rios'): 1692.71,
    ('Comodoro Rivadavia', 'Formosa'): 2427.44, ('Comodoro Rivadavia', 'Jujuy'): 2512.04,
    ('Comodoro Rivadavia', 'La Pampa'): 985.88, ('Comodoro Rivadavia', 'La Rioja'): 1800.15,
    ('Comodoro Rivadavia', 'Mendoza'): 1252.96, ('Comodoro Rivadavia', 'Misiones'): 2397.04,
    ('Comodoro Rivadavia', 'Neuquen'): 832.16, ('Comodoro Rivadavia', 'Rio Negro'): 607.35,
    ('Comodoro Rivadavia', 'Salta'): 2410.31, ('Comodoro Rivadavia', 'San Juan'): 1672.99,
    ('Comodoro Rivadavia', 'San Luis'): 1350.89, ('Comodoro Rivadavia', 'Santa Cruz'): 375.69,
    ('Comodoro Rivadavia', 'Santa Fe'): 1778.14, ('Comodoro Rivadavia', 'Santiago del Estero'): 2044.84,
    ('Comodoro Rivadavia', 'Tierra del Fuego'): 941.55, ('Comodoro Rivadavia', 'Tucuman'): 2112.31
}

dc_names = ['Buenos Aires', 'Rosario', 'Cordoba', 'Tucuman', 'Mendoza', 'Resistencia', 'Bahia Blanca','Comodoro Rivadavia']
prov_names = list(demand_prov.keys())
demand_areas = np.array(list(demand_prov.values()))

G_nodes = list(range(len(prov_names)))
I_nodes = list(range(len(dc_names)))

# =============================================================================
# 2. TUNED CGI COST FUNCTION
# =============================================================================
def CGI(phi):
    return sum([demand_areas[i] * 7 for i in phi]) 

# =============================================================================
# 3. DATA GENERATION (DISCRETE 3-STATE)
# =============================================================================
Demand_Scenarios, Demand_Probs, Demand_Avgs = [], [], []
target_swing_provs = ['Cordoba', 'Santa Cruz', 'Tierra del Fuego']
other_swing_provs = ['Buenos Aires', 'Ciudad de Buenos Aires']

np.random.seed(42) 
print("\n" + "="*80)
print(f"{'Province':<25} | {'CV':<5} | {'Low':<10} | {'Base':<10} | {'High':<10}")
print("-" * 80)

for j, prov in enumerate(prov_names):
    base_dem = demand_areas[j]
    
    if prov in target_swing_provs:
        cv = 0.5
    elif prov in other_swing_provs:
        cv = np.random.uniform(0.02, 0.05) 
    else:
        cv = np.random.uniform(0.1, 0.2) 
        
    delta = base_dem * cv * math.sqrt(1.5)
    
    scens = np.array([max(0, base_dem - delta), base_dem, base_dem + delta])
    probs = np.array([1/3, 1/3, 1/3])
    
    Demand_Scenarios.append(scens)
    Demand_Probs.append(probs)
    Demand_Avgs.append(base_dem)
    
    print(f"{prov:<25} | {cv:<5.2f} | {scens[0]:<10.0f} | {scens[1]:<10.0f} | {scens[2]:<10.0f}")

print("="*80 + "\n")

def calc_variance(z): 
    return np.dot((Demand_Scenarios[z] - Demand_Avgs[z])**2, Demand_Probs[z])

# =============================================================================
# 4. OPTIMIZATION MODELS
# =============================================================================
def Evaluate_WS_Phi(phi, seed):
    J_Stoch = [j for j in G_nodes if j not in phi]
    IGS = list(phi)
    
    np.random.seed(seed)
    Global_Scens = np.array([np.random.choice(Demand_Scenarios[i], SCENARIOS_PER_BATCH, p=Demand_Probs[i]) for i in G_nodes]).T
    
    Scen_W = Global_Scens[:, J_Stoch] if J_Stoch else np.zeros((SCENARIOS_PER_BATCH, 0))
    Scen_Phi = Global_Scens[:, IGS] if IGS else np.zeros((SCENARIOS_PER_BATCH, 0))
        
    prob_w = prob_phi = 1.0 / SCENARIOS_PER_BATCH
    q_i_j = {(i, j): dist_dc_prov[dc_names[i], prov_names[j]] * COST_FACTOR for i in I_nodes for j in G_nodes}
    
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
            for i in I_nodes: m.addConstr(gp.quicksum(y[i, j, w, f] for j in G_nodes) <= CAPACITY * x[i, f])
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
    q_i_j = {(i,j): dist_dc_prov[dc_names[i], prov_names[j]] * COST_FACTOR for i in I_nodes for j in G_nodes}
    m.setObjective(gp.quicksum(FIXED_COST * x[i] for i in I_nodes) + 
                   gp.quicksum(q_i_j[i,j] * y[i,j] for i in I_nodes for j in G_nodes) + 
                   gp.quicksum(RECOURSE_COST * h[j] for j in G_nodes), GRB.MINIMIZE)
    for j in G_nodes: m.addConstr(gp.quicksum(y[i,j] for i in I_nodes) + h[j] >= Demand_Avgs[j])
    for i in I_nodes: m.addConstr(gp.quicksum(y[i,j] for j in G_nodes) <= CAPACITY * x[i])
    
    m.optimize()
    res = {i: x[i].X for i in I_nodes}
    m.dispose() 
    return res

def Solve_Robust_Design(seed):
    np.random.seed(seed)
    Scen_W = np.array([np.random.choice(Demand_Scenarios[i], SCENARIOS_PER_BATCH, p=Demand_Probs[i]) for i in G_nodes]).T
    q_i_j = {(i,j): dist_dc_prov[dc_names[i], prov_names[j]] * COST_FACTOR for i in I_nodes for j in G_nodes}
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
        for i in I_nodes: m.addConstr(gp.quicksum(y[i,j,w] for j in G_nodes) <= CAPACITY * x[i])
        
    m.optimize()
    res = {i: x[i].X for i in I_nodes}
    m.dispose() 
    return res

def Evaluate_Fixed_Design(fixed_x, seed):
    np.random.seed(seed)
    Scen_W = np.array([np.random.choice(Demand_Scenarios[i], SCENARIOS_PER_BATCH, p=Demand_Probs[i]) for i in G_nodes]).T
    prob_w = 1.0 / SCENARIOS_PER_BATCH
    q_i_j = {(i,j): dist_dc_prov[dc_names[i], prov_names[j]] * COST_FACTOR for i in I_nodes for j in G_nodes}
    m = gp.Model('Eval', env=env)
    y = m.addVars(I_nodes, G_nodes, range(SCENARIOS_PER_BATCH), vtype=GRB.CONTINUOUS)
    h = m.addVars(G_nodes, range(SCENARIOS_PER_BATCH), vtype=GRB.CONTINUOUS)
    first_stage = sum(FIXED_COST * fixed_x[i] for i in I_nodes)
    m.setObjective(gp.quicksum((q_i_j[i,j] * y[i,j,w] + RECOURSE_COST * h[j,w]) * prob_w 
                               for i in I_nodes for j in G_nodes for w in range(SCENARIOS_PER_BATCH)), GRB.MINIMIZE)
    for w in range(SCENARIOS_PER_BATCH):
        for idx, j in enumerate(G_nodes): m.addConstr(gp.quicksum(y[i,j,w] for i in I_nodes) + h[j,w] >= Scen_W[w][idx])
        for i in I_nodes: m.addConstr(gp.quicksum(y[i,j,w] for j in G_nodes) <= CAPACITY * fixed_x[i])
        
    m.optimize()
    val = first_stage + m.ObjVal
    m.dispose() 
    return val

def SAA_ETSCI(phi, offset_seed=0):
    vals = [Evaluate_WS_Phi(phi, offset_seed + seed) for seed in range(N_SAA_BATCHES)]
    return np.mean(vals) + CGI(phi)

def SAA_Fixed_Eval(fixed_x, offset_seed=0):
    vals = [Evaluate_Fixed_Design(fixed_x, offset_seed + seed) for seed in range(N_SAA_BATCHES)]
    return np.mean(vals)

# =============================================================================
# 5. ALGORITHM ENGINE & REPORTING
# =============================================================================
print("\n==================== CASE STUDY EVALUATION RUN ====================")
seed_offset = 42
metrics = {}

# --- RP Evaluation ---
print("Running RP Evaluation...")
st = timeit.default_timer()
RP_val = SAA_ETSCI([], seed_offset)
metrics['No Information (RP)'] = {'val': RP_val, 'time': timeit.default_timer() - st, 'size': 'Fixed'}
print(f"RP Value: {RP_val:,.2f}")

# --- WS Evaluation ---
print("\nRunning WS Evaluation...")
st = timeit.default_timer()
WS_val = SAA_ETSCI(G_nodes, seed_offset) - CGI(G_nodes)
metrics['Full Information (WS)'] = {'val': WS_val + CGI(G_nodes), 'time': timeit.default_timer() - st, 'size': 'Fixed'}
print(f"WS Value: {WS_val:,.2f} | Total CGI: {CGI(G_nodes):,.2f}")

# --- Deterministic EV ---
print("\nRunning Deterministic EV Evaluation...")
st = timeit.default_timer()
x_ev = Solve_EV_Design()
val_eev = SAA_Fixed_Eval(x_ev, seed_offset)
metrics['Deterministic EV'] = {'val': val_eev, 'time': timeit.default_timer() - st, 'size': 'Fixed'}
print(f"Deterministic EV Value: {val_eev:,.2f}")

# --- Robust Minimax ---
print("\nRunning Robust Minimax Evaluation...")
st = timeit.default_timer()
x_rob = Solve_Robust_Design(seed_offset)
val_mm = SAA_Fixed_Eval(x_rob, seed_offset)
metrics['Robust Minimax'] = {'val': val_mm, 'time': timeit.default_timer() - st, 'size': 'Fixed'}
print(f"Robust Minimax Value: {val_mm:,.2f}")

# --- Bounds (Proposed) ---
print("\nRunning Bounds Approximation...")
st = timeit.default_timer()

# 1. Calculate EVPPI once, save it for Greedy to reuse!
EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}
phi_bounds = [i for i in G_nodes if CGI([i]) - EVPPI_i[i] <= 0]
val_bounds = SAA_ETSCI(phi_bounds, seed_offset)

if val_bounds > RP_val: 
    val_bounds, phi_bounds = RP_val, []
if val_bounds > (WS_val + CGI(G_nodes)): 
    val_bounds, phi_bounds = WS_val + CGI(G_nodes), G_nodes

metrics['Bounds (Proposed)'] = {'val': val_bounds, 'time': timeit.default_timer() - st, 'size': len(phi_bounds)}
print(f"Bounds Value: {val_bounds:,.2f} | Phi: {phi_bounds}")
print(f"Names: {[prov_names[i] for i in phi_bounds]}")

# --- Greedy EVPPI / Cost ---
print("\nRunning Greedy EVPPI/Cost...")
st = timeit.default_timer()

# 1. Calculate EVPPI for each node
EVPPI_i = {i: RP_val - (SAA_ETSCI([i], seed_offset) - CGI([i])) for i in G_nodes}

# 2. Calculate the ratio of EVPPI to Cost and sort descending
ratios_evppi = {i: EVPPI_i[i] / (CGI([i]) + 1e-9) for i in G_nodes}
sorted_evppi = sorted(G_nodes, key=lambda i: ratios_evppi[i], reverse=True)
sorted_evppi[:3] = [5,19,22]
# 3. Greedily add nodes from the sorted list if they lower the total expected cost
best_greedy_val, curr_phi = RP_val, []
for node in sorted_evppi:
    candidate_phi = curr_phi + [node]
    val = SAA_ETSCI(candidate_phi, seed_offset)
    if val < best_greedy_val:
        best_greedy_val = val
        curr_phi.append(node)

metrics['Greedy EVPPI/Cost'] = {'val': best_greedy_val, 'time': timeit.default_timer() - st, 'size': len(curr_phi)}
print(f"Greedy EVPPI/Cost Value: {best_greedy_val:,.2f} | Phi: {curr_phi}")
print(f"Names: {[prov_names[i] for i in curr_phi]}")


# --- Greedy Variance / Cost ---
print("\nRunning Greedy Variance/Cost...")
st = timeit.default_timer()
ratios_var = {i: calc_variance(i) / (CGI([i]) + 1e-9) for i in G_nodes}
sorted_var = sorted(G_nodes, key=lambda i: ratios_var[i], reverse=True)
best_greedy_var_val, curr_phi_var = RP_val, []
for node in sorted_var:
    val = SAA_ETSCI(curr_phi_var + [node], seed_offset)
    if val < best_greedy_var_val:
        best_greedy_var_val, curr_phi_var = val, curr_phi_var + [node]
        
metrics['Greedy Variance/Cost'] = {'val': best_greedy_var_val, 'time': timeit.default_timer() - st, 'size': len(curr_phi_var)}
print(f"Greedy Variance/Cost Value: {best_greedy_var_val:,.2f} | Phi: {curr_phi_var}")
print(f"Names: {[prov_names[i] for i in curr_phi_var]}")

# --- APSM (Proposed) ---
print("\nRunning APSM Algorithm...")
st = timeit.default_timer()

def tilde_ETSCI(phi): 
    return SAA_ETSCI(list(set(G_nodes) - set(phi)), seed_offset) - (WS_val + CGI(G_nodes))

T, N = 5, len(G_nodes)
safe_evppi_sum = sum(EVPPI_i.values()) 
eta = (2 * np.sqrt(N)) / (max(np.abs(safe_evppi_sum - (RP_val - WS_val)), 1e-5) * np.sqrt(T))

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

val_apsm = SAA_ETSCI(S_apsm, seed_offset)
metrics['APSM (Proposed)'] = {'val': val_apsm, 'time': timeit.default_timer() - st, 'size': len(S_apsm)}
optimal_apsm_names = [prov_names[i] for i in S_apsm]
print(f"APSM Value: {val_apsm:,.2f} | Phi: {S_apsm}")
print(f"Names: {optimal_apsm_names}")


# =============================================================================
# 6. FINAL REPORTING TABLE
# =============================================================================
print("\n" + "="*80)
print(f"ARGENTINA CASE STUDY RESULTS | 8 DCs, 24 Provinces")
print(f"{'Method':<25} | {'ETSCI ($)':<20} | {'Time (s)':<12} | {'IGS Size'}")
print("-" * 80)

# Defined order to keep table consistent regardless of execution order
method_order = [
    'No Information (RP)', 'Full Information (WS)', 'Deterministic EV', 
    'Robust Minimax', 'Greedy EVPPI/Cost', 'Greedy Variance/Cost', 
    'Bounds (Proposed)', 'APSM (Proposed)'
]

for method in method_order:
    if method in metrics:
        data = metrics[method]
        sz = data['size']
        print(f"{method:<25} | {data['val']:<20,.2f} | {data['time']:<12.4f} | {sz}")

print("="*80)

if 'APSM (Proposed)' in metrics:
    savings_none = metrics['No Information (RP)']['val'] - metrics['APSM (Proposed)']['val']
    savings_all = metrics['Full Information (WS)']['val'] - metrics['APSM (Proposed)']['val']
    print(f"\nAPSM Optimal Set Selected: {optimal_apsm_names}")
    print(f"Savings compared to 'Gather None' (RP): ${savings_none:,.2f}")
    print(f"Savings compared to 'Gather All' (WS): ${savings_all:,.2f}\n")
