import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import math

class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = [set() for _ in range(n)]

    def add_edge(self, u, v):
        if v not in self.adj[u]:
            self.adj[u].add(v)
            self.adj[v].add(u)

def mesh2d(p):
    s = int(round(math.sqrt(p)))
    G = Graph(p)
    def idx(i,j): return i*s+j
    for i in range(s):
        for j in range(s):
            u = idx(i,j)
            if i+1<s: G.add_edge(u, idx(i+1,j))
            if j+1<s: G.add_edge(u, idx(i,j+1))
    return G

def mesh3d(p):
    s = int(round(p**(1/3)))
    G = Graph(p)
    def idx(i,j,k): return i*s*s+j*s+k
    for i in range(s):
        for j in range(s):
            for k in range(s):
                u = idx(i,j,k)
                if i+1<s: G.add_edge(u, idx(i+1,j,k))
                if j+1<s: G.add_edge(u, idx(i,j+1,k))
                if k+1<s: G.add_edge(u, idx(i,j,k+1))
    return G

def hypercube_from_p(p):
    r = int(round(p**(1/6)))
    d = 2*r
    n = 1 << d
    G = Graph(n)
    for u in range(n):
        for i in range(d):
            v = u ^ (1<<i)
            if v > u:
                G.add_edge(u,v)
    return G, n

def add_random_edges(G,k,seed):
    random.seed(int(seed))
    for v in range(G.n):
        added=0
        while added<k:
            w=random.randrange(G.n)
            if w!=v and w not in G.adj[v]:
                G.add_edge(v,w)
                added+=1

def bfs(G,src):
    dist=[-1]*G.n
    dist[src]=0
    q=collections.deque([src])
    while q:
        u=q.popleft()
        for v in G.adj[u]:
            if dist[v]<0:
                dist[v]=dist[u]+1
                q.append(v)
    return dist

def estimate_diameter(G):
    s=random.randrange(G.n)
    d=bfs(G,s)
    far=d.index(max(d))
    d2=bfs(G,far)
    return max(d2)

def estimate_bisection(G,trials=20):
    best=10**18
    for _ in range(trials):
        perm=list(range(G.n))
        random.shuffle(perm)
        A=set(perm[:G.n//2])
        cut=0
        for u in A:
            for v in G.adj[u]:
                if v not in A:
                    cut+=1
        best=min(best,cut)
    return best

def estimate_congestion(G,trials=2000,seed=0):
    random.seed(int(seed))
    edge_load={}
    for _ in range(trials):
        s=random.randrange(G.n)
        t=random.randrange(G.n)
        if s==t: continue
        parent=[-1]*G.n
        dist=[-1]*G.n
        dist[s]=0
        q=collections.deque([s])
        while q:
            u=q.popleft()
            for v in G.adj[u]:
                if dist[v]<0:
                    dist[v]=dist[u]+1
                    parent[v]=u
                    q.append(v)
        cur=t
        while parent[cur]!=-1:
            u=parent[cur]
            e=(u,cur) if u<cur else (cur,u)
            edge_load[e]=edge_load.get(e,0)+1
            cur=u
    if not edge_load:
        return 0
    return max(edge_load.values())

ps=np.array([64,729,4096,15625,46656],dtype=int)
k=4

diam2d,diam3d,diamHC=[],[],[]
bis2d,bis3d,bisHC=[],[],[]
cong2d,cong3d,congHC=[],[],[]
hc_sizes=[]

for p in ps:

    # 2D
    G=mesh2d(p)
    add_random_edges(G,k,p)
    diam2d.append(estimate_diameter(G))
    bis2d.append(estimate_bisection(G))
    cong2d.append(estimate_congestion(G,seed=p))

    # 3D
    G=mesh3d(p)
    add_random_edges(G,k,p)
    diam3d.append(estimate_diameter(G))
    bis3d.append(estimate_bisection(G))
    cong3d.append(estimate_congestion(G,seed=p+1))

    # Hypercube
    G,n=hypercube_from_p(p)
    hc_sizes.append(n)
    add_random_edges(G,k,p)
    diamHC.append(estimate_diameter(G))
    bisHC.append(estimate_bisection(G))
    congHC.append(estimate_congestion(G,seed=p+2))

diam2d=np.array(diam2d)
diam3d=np.array(diam3d)
diamHC=np.array(diamHC)
bis2d=np.array(bis2d)
bis3d=np.array(bis3d)
bisHC=np.array(bisHC)
cong2d=np.array(cong2d)
cong3d=np.array(cong3d)
congHC=np.array(congHC)

# =====================================================
# Part 1: Diameter log fit
# =====================================================
coef2=np.polyfit(np.log(ps),diam2d,1)
coef3=np.polyfit(np.log(ps),diam3d,1)
coefH=np.polyfit(np.log(ps),diamHC,1)

# =====================================================
# Part 2: Bisection linear fit
# =====================================================
coefB2=np.polyfit(ps,bis2d,1)
coefB3=np.polyfit(ps,bis3d,1)
coefBH=np.polyfit(ps,bisHC,1)

# =====================================================
# Plotting
# =====================================================
plt.figure()
plt.scatter(ps,diam2d,label="2D")
plt.scatter(ps,diam3d,label="3D")
plt.scatter(ps,diamHC,label="Hypercube")
plt.xlabel("p")
plt.ylabel("Diameter")
plt.title("Diameter scaling")
plt.legend()
plt.show()

plt.figure()
plt.scatter(ps,bis2d,label="2D")
plt.scatter(ps,bis3d,label="3D")
plt.scatter(ps,bisHC,label="Hypercube")
plt.xlabel("p")
plt.ylabel("Bisection width")
plt.title("Bisection scaling")
plt.legend()
plt.show()

plt.figure()
plt.scatter(ps,cong2d,label="2D")
plt.scatter(ps,cong3d,label="3D")
plt.scatter(ps,congHC,label="Hypercube")
plt.xlabel("p")
plt.ylabel("Congestion (proxy)")
plt.title("Congestion scaling")
plt.legend()
plt.show()

print("=== Diameter log fits ===")
print("2D: D(p) ≈ {:.3f} ln(p) + {:.3f}".format(coef2[0],coef2[1]))
print("3D: D(p) ≈ {:.3f} ln(p) + {:.3f}".format(coef3[0],coef3[1]))
print("HC: D(p) ≈ {:.3f} ln(p) + {:.3f}".format(coefH[0],coefH[1]))

print("\n=== Bisection linear fits ===")
print("2D: B(p) ≈ {:.3f} p + {:.3f}".format(coefB2[0],coefB2[1]))
print("3D: B(p) ≈ {:.3f} p + {:.3f}".format(coefB3[0],coefB3[1]))
print("HC: B(p) ≈ {:.3f} p + {:.3f}".format(coefBH[0],coefBH[1]))

print("\nHypercube sizes used:",hc_sizes)

