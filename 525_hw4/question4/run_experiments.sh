#!/bin/bash
# CS 525 HW4 – Question 4 Experiment Runner (robust version)
# Assumes ./run1 (1D) and ./run2 (2D) are compiled in the current directory.
# Output: q4_part1.csv  q4_part2.csv  q4_part3.csv  q4_part4.csv
#
# Usage: bash run_experiments.sh

RUN1=./run1
RUN2=./run2

# --------------------------------------------------------------------------
# get_time  $1=executable  $2=p  $3=n
# Returns elapsed time in microseconds, or "ERR" on failure.
# --------------------------------------------------------------------------
get_time() {
    local exe=$1 p=$2 n=$3
    local output t

    output=$(mpirun -np "$p" "$exe" "$n" 2>/dev/null)
    if [[ $? -ne 0 ]]; then
        echo "ERR"
        return
    fi

    t=$(echo "$output" | grep -oP 'time=\K[0-9]+(\.[0-9]+)?')
    if [[ -z "$t" ]]; then
        echo "ERR"
    else
        echo "$t"
    fi
}

# --------------------------------------------------------------------------
# safe_div  $1=numerator  $2=denominator
# --------------------------------------------------------------------------
safe_div() {
    awk "BEGIN{printf \"%.4f\", $1 / $2}"
}

# ==========================================================================
# Part 1 – Fix n=2048, vary p = 1,2,4,8
# ==========================================================================
echo "========================================"
echo "Part 1: fixed n=2048, p in {1,2,4,8}"
echo "========================================"
OUT1=q4_part1.csv
echo "p,n,time1d_us,time2d_us,speedup1d,efficiency1d,speedup2d,efficiency2d" > "$OUT1"

N_FIXED=2048
printf "  Getting baselines (p=1, n=%d) ...\n" $N_FIXED
BASE1D_1=$(get_time "$RUN1" 1 $N_FIXED)
BASE2D_1=$(get_time "$RUN2" 1 $N_FIXED)
echo "  Baseline 1D=${BASE1D_1}us  2D=${BASE2D_1}us"
echo "1,$N_FIXED,$BASE1D_1,$BASE2D_1,1.0000,1.0000,1.0000,1.0000" >> "$OUT1"

for P in 1 2 3 4 5 6 7 8; do
    printf "  p=%-3s  n=%-6s  ... " "$P" "$N_FIXED"
    t1d=$(get_time "$RUN1" "$P" "$N_FIXED")
    t2d=$(get_time "$RUN2" "$P" "$N_FIXED")
    if [[ "$t1d" == "ERR" || "$t2d" == "ERR" || "$BASE1D_1" == "ERR" || "$BASE2D_1" == "ERR" ]]; then
        echo "SKIPPED (1D=$t1d  2D=$t2d)"
        echo "$P,$N_FIXED,ERR,ERR,ERR,ERR,ERR,ERR" >> "$OUT1"
        continue
    fi
    sp1d=$(safe_div "$BASE1D_1" "$t1d")
    sp2d=$(safe_div "$BASE2D_1" "$t2d")
    eff1d=$(safe_div "$sp1d" "$P")
    eff2d=$(safe_div "$sp2d" "$P")
    echo "1D=${t1d}us sp=${sp1d} eff=${eff1d} | 2D=${t2d}us sp=${sp2d} eff=${eff2d}"
    echo "$P,$N_FIXED,$t1d,$t2d,$sp1d,$eff1d,$sp2d,$eff2d" >> "$OUT1"
done
echo "  -> $OUT1"

# ==========================================================================
# Part 2 – Fix p=8, vary n = 2^9..2^12 (512, 1024, 2048, 4096)
# ==========================================================================
echo ""
echo "========================================"
echo "Part 2: fixed p=8, n in {512,1024,2048,4096}"
echo "========================================"
OUT2=q4_part2.csv
echo "n,p,time1d_us,time2d_us,speedup1d,efficiency1d,speedup2d,efficiency2d" > "$OUT2"

P_FIXED=8

for N in 512 1024 2048 4096; do
    printf "  Getting baseline p=1 for n=%d ...\n" $N
    B1=$(get_time "$RUN1" 1 "$N")
    B2=$(get_time "$RUN2" 1 "$N")
    echo "    Baseline 1D=${B1}us  2D=${B2}us"

    printf "  p=%-3s  n=%-6s  ... " "$P_FIXED" "$N"
    t1d=$(get_time "$RUN1" "$P_FIXED" "$N")
    t2d=$(get_time "$RUN2" "$P_FIXED" "$N")
    if [[ "$t1d" == "ERR" || "$t2d" == "ERR" || "$B1" == "ERR" || "$B2" == "ERR" ]]; then
        echo "SKIPPED (1D=$t1d  2D=$t2d)"
        echo "$N,$P_FIXED,ERR,ERR,ERR,ERR,ERR,ERR" >> "$OUT2"
        continue
    fi
    sp1d=$(safe_div "$B1" "$t1d")
    sp2d=$(safe_div "$B2" "$t2d")
    eff1d=$(safe_div "$sp1d" "$P_FIXED")
    eff2d=$(safe_div "$sp2d" "$P_FIXED")
    echo "1D=${t1d}us sp=${sp1d} eff=${eff1d} | 2D=${t2d}us sp=${sp2d} eff=${eff2d}"
    echo "$N,$P_FIXED,$t1d,$t2d,$sp1d,$eff1d,$sp2d,$eff2d" >> "$OUT2"
done
echo "  -> $OUT2"

# ==========================================================================
# Part 3 – Vary p = 1,2,4,8, n = 512*p
# ==========================================================================
echo ""
echo "========================================"
echo "Part 3: n=512*p, p in {1,2,4,8}"
echo "========================================"
OUT3=q4_part3.csv
echo "p,n,time1d_us,time2d_us,speedup1d,efficiency1d,speedup2d,efficiency2d" > "$OUT3"

printf "  Getting baseline p=1, n=512 ...\n"
BASE1D_3=$(get_time "$RUN1" 1 512)
BASE2D_3=$(get_time "$RUN2" 1 512)
echo "  Baseline 1D=${BASE1D_3}us  2D=${BASE2D_3}us"
echo "1,512,$BASE1D_3,$BASE2D_3,1.0000,1.0000,1.0000,1.0000" >> "$OUT3"

for P in 1 2 3 4 5 6 7 8; do
    N=$(( 512 * P ))
    printf "  p=%-3s  n=%-6s  ... " "$P" "$N"
    B1=$(get_time "$RUN1" 1 "$N")
    B2=$(get_time "$RUN2" 1 "$N")
    t1d=$(get_time "$RUN1" "$P" "$N")
    t2d=$(get_time "$RUN2" "$P" "$N")
    if [[ "$t1d" == "ERR" || "$t2d" == "ERR" || "$BASE1D_3" == "ERR" || "$BASE2D_3" == "ERR" ]]; then
        echo "SKIPPED (1D=$t1d  2D=$t2d)"
        echo "$P,$N,ERR,ERR,ERR,ERR,ERR,ERR" >> "$OUT3"
        continue
    fi
    sp1d=$(safe_div "$B1" "$t1d")
    sp2d=$(safe_div "$B2" "$t2d")
    eff1d=$(safe_div "$sp1d" "$P")
    eff2d=$(safe_div "$sp2d" "$P")
    echo "1D=${t1d}us sp=${sp1d} eff=${eff1d} | 2D=${t2d}us sp=${sp2d} eff=${eff2d}"
    echo "$P,$N,$t1d,$t2d,$sp1d,$eff1d,$sp2d,$eff2d" >> "$OUT3"
done
echo "  -> $OUT3"

# ==========================================================================
# Part 4 – Isoefficiency scaling
#
# 题目: For W = 1^2, ..., 8^2, run both programs on a matrix of size
#       (512*sqrt(W)) x (512*sqrt(W)).
#
# W = k^2  (k = 1..8)  =>  n = 512 * sqrt(W) = 512 * k
#
# 两种算法 isoefficiency 不同，相同 W 下所需 p 不同：
#   1D row-wise:  W = Theta(p^2)   => p = sqrt(W)    = k
#   2D block:     W = Theta(p^3/2) => p = W^(2/3)    = k^(4/3)  (ceiling)
# ==========================================================================
echo ""
echo "========================================"
echo "Part 4: Isoefficiency, W = k^2, k in {1..8}"
echo "========================================"
OUT4=q4_part4.csv
echo "k,W,n,p_1d,p_2d,time1d_us,speedup1d,efficiency1d,time2d_us,speedup2d,efficiency2d" > "$OUT4"

# Helper: solve p * (ln p)^2 ≈ W, return the closest integer p
# 修正后的求解器：寻找满足 p * (log2 p)^2 >= W 的最小整数 p (Ceiling logic)
solve_p2d() {
    local W="$1"
    python3 - "$W" <<'PY'
import math, sys
W = float(sys.argv[1])
if W <= 1.0:
    print(1)
    sys.exit(0)

# 线性搜索满足等效率约束的最小 p 值
p = 1
while True:
    p += 1
    # 2D 算法的开销项主要受 p * (log p)^2 驱动
    if p * (math.log2(p)**2) >= W:
        print(p)
        break
    if p > 2048: # 安全限制
        print(p)
        break
PY
}

# Baseline: k=1 => W=1 => n=512, p=1 for both
printf "  Getting baseline k=1, W=1, n=512, p=1 for both ...\n"
BASE1D_4=$(get_time "$RUN1" 1 512)
BASE2D_4=$(get_time "$RUN2" 1 512)
echo "  Baseline 1D=${BASE1D_4}us  2D=${BASE2D_4}us"
echo "1,1,512,1,1,$BASE1D_4,1.0000,1.0000,$BASE2D_4,1.0000,1.0000" >> "$OUT4"

for k in 2 3 4 5 6 7 8; do
    W=$(( k * k ))
    N=$(( 512 * k ))

    # 1D isoefficiency: W = Θ(p^2) => p = sqrt(W) = k
    P1D=$k

    # 2D isoefficiency: W = Θ(p log^2 p)
    # Solve numerically for the closest integer p
    P2D=$(solve_p2d "$W")

    printf "  k=%-2s  W=%-4s  n=%-6s  p_1d=%-4s  p_2d=%-4s\n" \
           "$k" "$W" "$N" "$P1D" "$P2D"
    B1=$(get_time "$RUN1" 1 "$N")   # T(1, n=512*k)
    B2=$(get_time "$RUN2" 1 "$N")
    t1d=$(get_time "$RUN1" "$P1D" "$N")
    t2d=$(get_time "$RUN2" "$P2D" "$N")

    if [[ "$t1d" == "ERR" || "$t2d" == "ERR" || \
          "$BASE1D_4" == "ERR" || "$BASE2D_4" == "ERR" ]]; then
        echo "    SKIPPED (1D=$t1d  2D=$t2d)"
        echo "$k,$W,$N,$P1D,$P2D,ERR,ERR,ERR,ERR,ERR,ERR" >> "$OUT4"
        continue
    fi

    sp1d=$(safe_div "$B1" "$t1d")
    sp2d=$(safe_div "$B2" "$t2d")
    eff1d=$(safe_div "$sp1d" "$P1D")
    eff2d=$(safe_div "$sp2d" "$P2D")

    echo "    1D: ${t1d}us  speedup=${sp1d}  eff=${eff1d}"
    echo "    2D: ${t2d}us  speedup=${sp2d}  eff=${eff2d}"
    echo "$k,$W,$N,$P1D,$P2D,$t1d,$sp1d,$eff1d,$t2d,$sp2d,$eff2d" >> "$OUT4"
done

echo "  -> $OUT4"
 
# ==========================================================================
echo ""
echo "All done. Output files:"
echo "  $OUT1  – Part 1: fixed n=2048, vary p"
echo "  $OUT2  – Part 2: fixed p=8,   vary n"
echo "  $OUT3  – Part 3: n=512*p,     vary p"
echo "  $OUT4  – Part 4: isoefficiency (W=k^2, k=1..8)"

#chmod +x run_experiments.sh
#bash run_experiments.sh