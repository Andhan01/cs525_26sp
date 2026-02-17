import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from scipy.optimize import curve_fit

def power_law(p, a, b):
    return a * (p ** b)

def bfs(adj, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        v = q.popleft()
        for w in adj[v]:
            if w not in dist:
                dist[w] = dist[v] + 1
                q.append(w)
    return dist

def add_random_edges(adj, k):
    random.seed(0)
    nodes = list(adj.keys())

    for v in nodes:
        k_prime = k

        while k_prime != 0:

            candidates = random.sample(
                [x for x in nodes if x != v],
                k_prime
            )

            E_prime = []
            for w in candidates:
                if w not in adj[v]:
                    E_prime.append(w)

            for w in E_prime:
                adj[v].add(w)
                adj[w].add(v)
            k_prime -= len(E_prime)
    return adj
def build_2d_mesh(p):
    n = int(math.sqrt(p))
    adj = {i: set() for i in range(p)}

    for r in range(n):
        for c in range(n):
            idx = r * n + c
            if r + 1 < n:
                adj[idx].add((r+1)*n + c)
                adj[(r+1)*n + c].add(idx)
            if c + 1 < n:
                adj[idx].add(r*n + (c+1))
                adj[r*n + (c+1)].add(idx)

    return adj
def build_3d_mesh(p):
    n = round(p ** (1/3))
    adj = {i: set() for i in range(p)}

    for x in range(n):
        for y in range(n):
            for z in range(n):
                idx = x*n*n + y*n + z
                if x+1 < n:
                    nid = (x+1)*n*n + y*n + z
                    adj[idx].add(nid)
                    adj[nid].add(idx)
                if y+1 < n:
                    nid = x*n*n + (y+1)*n + z
                    adj[idx].add(nid)
                    adj[nid].add(idx)
                if z+1 < n:
                    nid = x*n*n + y*n + (z+1)
                    adj[idx].add(nid)
                    adj[nid].add(idx)

    return adj

def build_hypercube(d):
    # d = int(math.log2(p))
    p = 2**d
    adj = {i: set() for i in range(p)}

    for i in range(p):
        for bit in range(d):
            neighbor = i ^ (1 << bit)
            adj[i].add(neighbor)

    return adj

def estimate_diameter(adj, samples=1000):
    nodes = list(adj.keys())
    max_d = 0

    for _ in range(samples):
        src = random.choice(nodes)
        dist = bfs(adj, src)
        max_d = max(max_d, max(dist.values()))

    return max_d

def estimate_bisection(adj, trials=1000):
    nodes = list(adj.keys())
    p = len(nodes)
    min_cut = float("inf")

    for _ in range(trials):
        part = set(random.sample(nodes, p//2))
        cut = 0
        for v in part:
            for w in adj[v]:
                if w not in part:
                    cut += 1
        min_cut = min(min_cut, cut)

    return min_cut

def compute_dilation_and_congestion(adjA, adjB):
    p = len(adjA)
    max_dilation = 0
    edge_load = defaultdict(int)

    for u in range(p):
        neighbors_in_A = [v for v in adjA[u] if v > u]
        if not neighbors_in_A:
            continue

        dist = {u: 0}
        parent = {u: None}
        q = deque([u])

        found_count = 0
        target_count = len(neighbors_in_A)

        while q:
            curr = q.popleft()

            if curr in neighbors_in_A:
                found_count += 1

            if found_count == target_count:
                break

            for nxt in adjB[curr]:
                if nxt not in dist:
                    dist[nxt] = dist[curr] + 1
                    parent[nxt] = curr
                    q.append(nxt)


        for v in neighbors_in_A:
            if v not in dist:
                continue

            max_dilation = max(max_dilation, dist[v])

            curr = v
            while curr != u:
                p_node = parent[curr]
                edge = tuple(sorted((curr, p_node)))
                edge_load[edge] += 1
                curr = p_node

    max_congestion = max(edge_load.values()) if edge_load else 0
    return max_dilation, max_congestion


def compute_average_distance(adj, samples=500):
    nodes = list(adj.keys())
    total_dist = 0
    count = 0

    for _ in range(samples):
        src = random.choice(nodes)
        dst = random.choice(nodes)
        if src == dst:
            continue

        dist = bfs(adj, src)
        if dst in dist:
            total_dist += dist[dst]
            count += 1

    return total_dist / count if count > 0 else float('inf')


def compare_networks(adjA, adjB, link_speed_A, link_speed_B, p, k):
    print(f"\n{'='*60}")
    print(f"Network Comparison for p={p}, k={k}")
    print(f"{'='*60}")


    avg_dist_A = compute_average_distance(adjA, samples=500)
    avg_dist_B = compute_average_distance(adjB, samples=500)

    diam_A = estimate_diameter(adjA, samples=300)
    diam_B = estimate_diameter(adjB, samples=300)

    print(f"\nNetwork (a) - 2D Mesh + {k} random edges:")
    print(f"  - Average distance: {avg_dist_A:.2f} hops")
    print(f"  - Diameter: {diam_A} hops")
    print(f"  - Link speed: {link_speed_A} Mb/s")

    print(f"\nNetwork (b) - 3D Mesh + {k} random edges:")
    print(f"  - Average distance: {avg_dist_B:.2f} hops")
    print(f"  - Diameter: {diam_B} hops")
    print(f"  - Link speed: {link_speed_B} Mb/s")

    perf_metric_A = avg_dist_A / link_speed_A
    perf_metric_B = avg_dist_B / link_speed_B

    print(f"  - Network (a): {perf_metric_A:.6f}")
    print(f"  - Network (b): {perf_metric_B:.6f}")

    # 比较
    if perf_metric_A < perf_metric_B:
        speedup = perf_metric_B / perf_metric_A
        print(f"  - Network (a) is {speedup:.2f}x faster than Network (b)")
    else:
        speedup = perf_metric_A / perf_metric_B
        print(f"  - Network (b) is {speedup:.2f}x faster than Network (a)")

    print(f"\nAdditional Analysis:")
    print(f"  - Network (a) has {link_speed_A/link_speed_B:.1f}x higher link bandwidth")
    print(f"  - Network (b) has {avg_dist_A/avg_dist_B:.2f}x shorter average path")

    return perf_metric_A, perf_metric_B




# ============================================================
# Main Function
# ============================================================

# p_values = [2**6, 3**6, 4**6, 5**6, 6**6]
p_values = [2**6, 3**6, 4**6]
k = 4


diam_2d = []
diam_3d = []
diam_hc = []

bisec_2d = []
bisec_3d = []
bisec_hc = []


dilation_lst = []
congestion_lst = []


for p in p_values:
    print(f"Running p = {p}")
    #============================Q1 and Q2==============================
    # 2D
    adj2d = build_2d_mesh(p)
    adj2d = add_random_edges(adj2d, k)
    diam_2d.append(estimate_diameter(adj2d))
    bisec_2d.append(estimate_bisection(adj2d))
    # 3D
    adj3d = build_3d_mesh(p)
    adj3d = add_random_edges(adj3d, k)
    diam_3d.append(estimate_diameter(adj3d))
    bisec_3d.append(estimate_bisection(adj3d))

    dim_hc = int(2 * (p**(1/6)))
    num_nodes_hc = 2**dim_hc
    adjhc = build_hypercube(dim_hc)
    adjhc = add_random_edges(adjhc, k)
    diam_hc.append(estimate_diameter(adjhc))
    bisec_hc.append(estimate_bisection(adjhc))
    
    #===============================Q3===================================
    dilation, congestion = compute_dilation_and_congestion(adjA=adj2d, adjB=adj3d)
    dilation_lst.append(dilation)
    congestion_lst.append(congestion)
    print(f"p = {p}: Dilation = {dilation}, Congestion = {congestion}")



#=====================================plot of Q1=====================================

p_fit = np.linspace(min(p_values), max(p_values), 100)



params_2d, _ = curve_fit(power_law, p_values, diam_2d)
params_3d, _ = curve_fit(power_law, p_values, diam_3d)
params_hc, _ = curve_fit(power_law, p_values, diam_hc)

plt.plot(p_fit, power_law(p_fit, *params_2d), '--', label=f"2D fit: {params_2d[0]:.2f}*p^{params_2d[1]:.3f}")
plt.plot(p_fit, power_law(p_fit, *params_3d), '--', label=f"3D fit: {params_3d[0]:.2f}*p^{params_3d[1]:.3f}")
plt.plot(p_fit, power_law(p_fit, *params_hc), '--', label=f"hc fit: {params_hc[0]:.2f}*p^{params_hc[1]:.3f}")



plt.plot(p_values, diam_2d,'s-', label="2D Mesh + k")
plt.plot(p_values, diam_3d, 'o-', label="3D Mesh + k")
plt.plot(p_values, diam_hc, '^-', label="Hypercube + k")
plt.xlabel("p")
plt.ylabel("Estimated Diameter")
plt.legend()
plt.show()

# print('bisection of 2d-mesh ',bisec_2d)
# print('bisection of 3d-mesh ',bisec_3d)
# print('bisection of hypercube ',bisec_hc)


#=====================================plot of Q2=====================================

paramss_2d, _ = curve_fit(power_law, p_values, bisec_2d)
paramss_3d, _ = curve_fit(power_law, p_values, bisec_3d)
paramss_hc, _ = curve_fit(power_law, p_values, bisec_hc)

plt.plot(p_fit, power_law(p_fit, *paramss_2d), '--', label=f"2D fit: {paramss_2d[0]:.2f}*p^{paramss_2d[1]:.3f}")
plt.plot(p_fit, power_law(p_fit, *paramss_3d), '--', label=f"3D fit: {paramss_3d[0]:.2f}*p^{paramss_3d[1]:.3f}")
plt.plot(p_fit, power_law(p_fit, *paramss_hc), '--', label=f"hc fit: {paramss_hc[0]:.2f}*p^{paramss_hc[1]:.3f}")



plt.plot(p_values, bisec_2d,'s-', label="2D Mesh + k")
plt.plot(p_values, bisec_3d, 'o-', label="3D Mesh + k")
plt.plot(p_values, bisec_hc, '^-', label="Hypercube + k")
plt.xlabel("p")
plt.ylabel("Estimated Bisection")
plt.legend()
plt.show()


#=====================================plot of Q3=====================================

plt.plot(p_values, dilation_lst, 'o-', label="dilation vs p")
plt.xlabel("p")
plt.ylabel("Estimated Dilation")
plt.legend()
plt.show()
plt.plot(p_values, congestion_lst, 'o-', label="congestion vs p")
plt.xlabel("p")
plt.ylabel("Estimated Congestion")
plt.legend()
plt.show()

#=====================================output of Q4=====================================


print("\n" + "="*70)
print("Question 4: Network Performance Comparison")
print("="*70)

for para4 in [(2**6, 4),(4**6, 2)]:
    print(f"\n>>> Q4 p= {para4[0]}, k = {para4[1]}")
    p1, k1 = para4[0], para4[1]

    adj2d_q4 = build_2d_mesh(p1)
    adj2d_q4 = add_random_edges(adj2d_q4, k1)

    adj3d_q4 = build_3d_mesh(p1)
    adj3d_q4 = add_random_edges(adj3d_q4, k1)

    compare_networks(
        adjA=adj2d_q4,
        adjB=adj3d_q4,
        link_speed_A=500,
        link_speed_B=200,
        p=p1,
        k=k1
    )


# Please simulate the network to estimate the parameters. 
# You need to compute the number of links in E' 
# that any edge in E is mapped onto and then took the max count to estimate te dilation of the mapping



# I did 10k sampling times to get the bisection. You can choose a reasonable number of iteration times. 
# If there is high variance, please consider increasing the sampling times or report the minimum/median number. 
# But you need to write the report clearly about your data.

# Fit a curve with 5 data points. There are lots of curve fitting functions available, 
# you are free to choose any based on your choice of tool/language. 
# You can fit the curve and then match against theoretical one.
# For part 2, please fit the curve too.
# For part 3, yes, please include plots and any relevant informations.


# Since no explicit strategy is given, like for Parts 1 and 2, can we sample src, dest to approximate both dilation and congestion?
# You need to simulate the network and compute by deriving the values for each edge and take the max.


#Q6 Part 3 Use the identity mapping as specified in the problem statement.


