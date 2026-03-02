/*
 * CS 52500 Spring 2026 — Homework #3, Question 3
 * Performance Comparison: LBST vs pBST
 *
 * All code from Q1 (LBST) and Q2 (pBST) is copied verbatim here,
 * as required by the submission guidelines ("copy the code you want
 * to re-use").
 *
 * Q3 adds:
 *   - Algorithm 3: random operation sequence generator
 *   - gettimeofday() timing for LBST and pBST
 *   - True thread-pool for pBST (MAX_THREAD_COUNT = 1 .. 128)
 */

#define _GNU_SOURCE

#include <assert.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/* ══════════════════════════════════════════════════════════════════════
 * §1  Shared node definition
 *     (Q1 fields + Q2 rwlock, so one struct serves both trees)
 * ══════════════════════════════════════════════════════════════════════ */
typedef struct vertex {
    int value;
    int deleted;
    struct vertex *left;
    struct vertex *right;
    int size;        /* LBST only */
    int n_marked;    /* LBST only */
    pthread_rwlock_t lock;  /* pBST only */
} vertex_t;

/* ══════════════════════════════════════════════════════════════════════
 * §2  Starter-code helpers (copied from Q1 / Q2, one-to-one)
 * ══════════════════════════════════════════════════════════════════════ */

int randint(int lower, int upper) {
    return (lower + rand() % (upper - lower));
}

/* create_vertex initialises ALL fields so the node works for both trees */
vertex_t *create_vertex(int value) {
    vertex_t *vet = (vertex_t *)malloc(sizeof(vertex_t));
    assert(vet != NULL);
    vet->value    = value;
    vet->deleted  = 0;
    vet->left     = NULL;
    vet->right    = NULL;
    vet->size     = 1;
    vet->n_marked = 0;
    pthread_rwlock_init(&vet->lock, NULL);
    return vet;
}

void free_tree(vertex_t **root) {
    if (root == NULL || *root == NULL)
        return;
    free_tree(&(*root)->left);
    free_tree(&(*root)->right);
    pthread_rwlock_destroy(&(*root)->lock);
    free(*root);
    *root = NULL;
}

void inorder_traversal_tree(vertex_t *root) {
    if (root == NULL) return;
    inorder_traversal_tree(root->left);
    if (root->deleted != 1)
        printf("%d, ", root->value);
    inorder_traversal_tree(root->right);
}

void print_tree(vertex_t *root, int level) {
    if (root == NULL) return;
    printf("|");
    for (int i = 0; i < level; i++) printf("-");
    printf(" %d", root->value);
    if (root->deleted == 1) printf(" (deleted)");
    printf("\n");
    print_tree(root->left,  level + 1);
    print_tree(root->right, level + 1);
}

/* ══════════════════════════════════════════════════════════════════════
 * §3  LBST internals (copied from Q1)
 * ══════════════════════════════════════════════════════════════════════ */

static int sz(vertex_t *v) { return v ? v->size     : 0; }
static int nm(vertex_t *v) { return v ? v->n_marked : 0; }

static void update(vertex_t *v) {
    if (!v) return;
    v->size     = 1 + sz(v->left) + sz(v->right);
    v->n_marked = v->deleted + nm(v->left) + nm(v->right);
}

typedef struct { vertex_t **data; int size, capacity; } node_array_t;

static void array_init(node_array_t *a) {
    a->size = 0; a->capacity = 16;
    a->data = malloc(a->capacity * sizeof(vertex_t *)); assert(a->data);
}
static void array_push(node_array_t *a, vertex_t *v) {
    if (a->size >= a->capacity) {
        a->capacity *= 2;
        a->data = realloc(a->data, a->capacity * sizeof(vertex_t *));
        assert(a->data);
    }
    a->data[a->size++] = v;
}
static void array_free_struct(node_array_t *a) { free(a->data); }

static void collect_inorder(vertex_t *root, node_array_t *out, int skip_marked) {
    if (!root) return;
    collect_inorder(root->left, out, skip_marked);
    if (!skip_marked || !root->deleted) array_push(out, root);
    collect_inorder(root->right, out, skip_marked);
}

static vertex_t *build_balanced(vertex_t **nodes, int l, int r) {
    if (l > r) return NULL;
    int mid = (l + r) / 2;
    vertex_t *root = nodes[mid];
    root->left  = build_balanced(nodes, l, mid - 1);
    root->right = build_balanced(nodes, mid + 1, r);
    update(root);
    return root;
}

static vertex_t *rebalance_insert(vertex_t *vr) {
    node_array_t arr; array_init(&arr);
    collect_inorder(vr, &arr, 0);
    for (int i = 0; i < arr.size; i++)
        arr.data[i]->left = arr.data[i]->right = NULL;
    vertex_t *nr = build_balanced(arr.data, 0, arr.size - 1);
    array_free_struct(&arr);
    return nr;
}

static vertex_t *rebalance_delete(vertex_t *vr) {
    node_array_t all, live; array_init(&all); array_init(&live);
    collect_inorder(vr, &all, 0);
    for (int i = 0; i < all.size; i++) {
        vertex_t *n = all.data[i];
        n->left = n->right = NULL;
        if (!n->deleted) array_push(&live, n);
        else { pthread_rwlock_destroy(&n->lock); free(n); }
    }
    vertex_t *nr = (live.size > 0)
                   ? build_balanced(live.data, 0, live.size - 1) : NULL;
    array_free_struct(&all); array_free_struct(&live);
    return nr;
}

vertex_t *lazy_insert(vertex_t *vr, int val) {
    if (!vr) return create_vertex(val);
    if (vr->value >= val) vr->left  = lazy_insert(vr->left,  val);
    else                  vr->right = lazy_insert(vr->right, val);
    update(vr);
    int diff = sz(vr->left) - sz(vr->right);
    if (diff < 0) diff = -diff;
    if (3 * diff >= vr->size) vr = rebalance_insert(vr);
    return vr;
}

vertex_t *lazy_delete(vertex_t *vr, int val) {
    if (!vr) return NULL;
    if      (vr->value > val) vr->left  = lazy_delete(vr->left,  val);
    else if (vr->value < val) vr->right = lazy_delete(vr->right, val);
    else                      vr->deleted = 1;
    update(vr);
    if (vr->n_marked * 2 > vr->size) vr = rebalance_delete(vr);
    return vr;
}

int look_up(vertex_t *vr, int val) {
    if (!vr) return 0;
    if (vr->value == val) return !vr->deleted;
    if (vr->value  > val) return look_up(vr->left,  val);
    return                       look_up(vr->right, val);
}

/* ══════════════════════════════════════════════════════════════════════
 * §4  pBST internals (copied from Q2)
 * ══════════════════════════════════════════════════════════════════════ */

static vertex_t        *bst_root    = NULL;
static pthread_rwlock_t root_rwlock = PTHREAD_RWLOCK_INITIALIZER;

void pBST_insert(int val) {
    pthread_rwlock_wrlock(&root_rwlock);
    if (!bst_root) {
        bst_root = create_vertex(val);
        pthread_rwlock_unlock(&root_rwlock);
        return;
    }
    vertex_t *curr = bst_root;
    pthread_rwlock_wrlock(&curr->lock);
    pthread_rwlock_unlock(&root_rwlock);
    while (1) {
        vertex_t **slot = (curr->value >= val) ? &curr->left : &curr->right;
        vertex_t  *next = *slot;
        if (!next) { *slot = create_vertex(val); pthread_rwlock_unlock(&curr->lock); return; }
        pthread_rwlock_wrlock(&next->lock);
        pthread_rwlock_unlock(&curr->lock);
        curr = next;
    }
}

void pBST_delete(int val) {
    pthread_rwlock_wrlock(&root_rwlock);
    if (!bst_root) { pthread_rwlock_unlock(&root_rwlock); return; }
    pthread_rwlock_t  *parent_lock = &root_rwlock;
    vertex_t         **child_ptr   = &bst_root;
    vertex_t          *curr        = bst_root;
    pthread_rwlock_wrlock(&curr->lock);
    while (curr->value != val) {
        vertex_t **next_slot = (curr->value > val) ? &curr->left : &curr->right;
        vertex_t  *next      = *next_slot;
        if (!next) { pthread_rwlock_unlock(parent_lock); pthread_rwlock_unlock(&curr->lock); return; }
        pthread_rwlock_wrlock(&next->lock);
        pthread_rwlock_unlock(parent_lock);
        parent_lock = &curr->lock; child_ptr = next_slot; curr = next;
    }
    if (!curr->left && !curr->right) {
        *child_ptr = NULL;
        pthread_rwlock_unlock(parent_lock); pthread_rwlock_unlock(&curr->lock);
        pthread_rwlock_destroy(&curr->lock); free(curr);
    } else if (!curr->left) {
        *child_ptr = curr->right; curr->right = NULL;
        pthread_rwlock_unlock(parent_lock); pthread_rwlock_unlock(&curr->lock);
        pthread_rwlock_destroy(&curr->lock); free(curr);
    } else if (!curr->right) {
        *child_ptr = curr->left; curr->left = NULL;
        pthread_rwlock_unlock(parent_lock); pthread_rwlock_unlock(&curr->lock);
        pthread_rwlock_destroy(&curr->lock); free(curr);
    } else {
        vertex_t *succ_par = curr, *succ = curr->right;
        pthread_rwlock_wrlock(&succ->lock);
        while (succ->left) {
            vertex_t *next = succ->left;
            pthread_rwlock_wrlock(&next->lock);
            if (succ_par != curr) pthread_rwlock_unlock(&succ_par->lock);
            succ_par = succ; succ = next;
        }
        curr->value = succ->value;
        if (succ_par == curr) curr->right    = succ->right;
        else { succ_par->left = succ->right; pthread_rwlock_unlock(&succ_par->lock); }
        pthread_rwlock_unlock(&succ->lock);
        succ->left = succ->right = NULL;
        pthread_rwlock_destroy(&succ->lock); free(succ);
        pthread_rwlock_unlock(parent_lock); pthread_rwlock_unlock(&curr->lock);
    }
}

bool pBST_lookup(int val) {
    pthread_rwlock_rdlock(&root_rwlock);
    vertex_t *curr = bst_root;
    if (!curr) { pthread_rwlock_unlock(&root_rwlock); return false; }
    pthread_rwlock_rdlock(&curr->lock);
    pthread_rwlock_unlock(&root_rwlock);
    while (curr) {
        if (curr->value == val) { pthread_rwlock_unlock(&curr->lock); return true; }
        vertex_t *next = (curr->value > val) ? curr->left : curr->right;
        if (!next) { pthread_rwlock_unlock(&curr->lock); return false; }
        pthread_rwlock_rdlock(&next->lock);
        pthread_rwlock_unlock(&curr->lock);
        curr = next;
    }
    return false;
}

/* ══════════════════════════════════════════════════════════════════════
 * §5  Operation sequence types
 * ══════════════════════════════════════════════════════════════════════ */

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_LOOKUP 2

typedef struct { int type; int val; } Op;

/* ══════════════════════════════════════════════════════════════════════
 * §6  Algorithm 3: Random Sequence of Operations
 *
 *   m = 3 * 10^6
 *   Z1 = ceil(m/3)   distinct random integers from Z_m
 *   Z2 = ceil(|Z1|/2) distinct random integers from Z1
 *   Z3 = ceil(m/2)   random integers from Z_m (may repeat)
 *   D  = D1 + D2 + D3, then shuffled
 *
 * Uses srand(0) from main — caller must call srand(0) before generate_D().
 * ══════════════════════════════════════════════════════════════════════ */
static Op *generate_D(int *out_size) {
    const int m      = 3000000;
    const int z1_sz  = (m + 2) / 3;          /* ceil(m/3)       = 1 000 000 */
    const int z2_sz  = (z1_sz + 1) / 2;      /* ceil(|Z1|/2)    =   500 000 */
    const int z3_sz  = (m + 1) / 2;          /* ceil(m/2)       = 1 500 000 */
    const int D_sz   = z1_sz + z2_sz + z3_sz; /* total ops       = 3 000 000 */

    /* ── build Z1: Fisher-Yates partial shuffle on [0..m-1] ── */
    int *pool = malloc(m * sizeof(int));
    assert(pool);
    for (int i = 0; i < m; i++) pool[i] = i;
    for (int i = 0; i < z1_sz; i++) {
        int j = i + rand() % (m - i);
        int t = pool[i]; pool[i] = pool[j]; pool[j] = t;
    }
    /* pool[0..z1_sz-1] is Z1 */

    /* ── build Z2: Fisher-Yates partial shuffle on Z1 ── */
    for (int i = 0; i < z2_sz; i++) {
        int j = i + rand() % (z1_sz - i);
        int t = pool[i]; pool[i] = pool[j]; pool[j] = t;
    }
    /* pool[0..z2_sz-1] is Z2,  pool[0..z1_sz-1] is still Z1 */

    /* ── assemble D1, D2, D3 ── */
    Op *D = malloc(D_sz * sizeof(Op));
    assert(D);

    /* D1: insert every element of Z1 */
    for (int i = 0; i < z1_sz; i++) {
        D[i].type = OP_INSERT;
        D[i].val  = pool[i];
    }

    /* D2: delete every element of Z2 */
    for (int i = 0; i < z2_sz; i++) {
        D[z1_sz + i].type = OP_DELETE;
        D[z1_sz + i].val  = pool[i];  /* Z2 is pool[0..z2_sz-1] */
    }

    /* D3: for each z in Z3, sample p ~ uniform[0,1] */
    for (int i = 0; i < z3_sz; i++) {
        int   z = rand() % m;
        double p = (double)rand() / (double)RAND_MAX;
        int idx  = z1_sz + z2_sz + i;
        D[idx].val  = z;
        D[idx].type = (p >= 0.5) ? OP_LOOKUP : OP_DELETE;
    }

    free(pool);

    /* ── shuffle D (Fisher-Yates) ── */
    for (int i = 0; i < D_sz - 1; i++) {
        int j = i + rand() % (D_sz - i);
        Op t  = D[i]; D[i] = D[j]; D[j] = t;
    }

    *out_size = D_sz;
    return D;
}

/* ══════════════════════════════════════════════════════════════════════
 * §7  Timing helper
 * ══════════════════════════════════════════════════════════════════════ */
static double elapsed_ms(struct timeval *start, struct timeval *end) {
    return (end->tv_sec  - start->tv_sec)  * 1000.0
         + (end->tv_usec - start->tv_usec) / 1000.0;
}

/* ══════════════════════════════════════════════════════════════════════
 * §8  LBST benchmark: run D sequentially on a fresh LBST
 * ══════════════════════════════════════════════════════════════════════ */
static double bench_lbst(Op *D, int D_size) {
    vertex_t *root = NULL;
    struct timeval t0, t1;

    gettimeofday(&t0, NULL);
    for (int i = 0; i < D_size; i++) {
        if      (D[i].type == OP_INSERT) root = lazy_insert(root, D[i].val);
        else if (D[i].type == OP_DELETE) root = lazy_delete(root, D[i].val);
        else                             look_up(root, D[i].val);
    }
    gettimeofday(&t1, NULL);

    free_tree(&root);
    return elapsed_ms(&t0, &t1);
}

/* ══════════════════════════════════════════════════════════════════════
 * §9  pBST thread-pool benchmark
 *
 * Unlike Q2's serial dispatch(), here we launch `thread_count` workers
 * that ALL run concurrently, each claiming the next unprocessed op.
 * This gives true parallelism and is what the performance plot measures.
 * ══════════════════════════════════════════════════════════════════════ */
static Op          *g_D;
static int          g_D_size;
static int          g_next_op;
static pthread_mutex_t g_op_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *pool_worker(void *arg) {
    (void)arg;
    while (1) {
        pthread_mutex_lock(&g_op_mutex);
        int idx = g_next_op++;
        pthread_mutex_unlock(&g_op_mutex);

        if (idx >= g_D_size) break;

        if      (g_D[idx].type == OP_INSERT) pBST_insert(g_D[idx].val);
        else if (g_D[idx].type == OP_DELETE) pBST_delete(g_D[idx].val);
        else                                  pBST_lookup(g_D[idx].val);
    }
    return NULL;
}

static double bench_pbst(Op *D, int D_size, int thread_count) {
    /* Reset the pBST */
    free_tree(&bst_root);
    bst_root = NULL;

    g_D       = D;
    g_D_size  = D_size;
    g_next_op = 0;

    struct timeval t0, t1;
    pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    assert(threads);

    gettimeofday(&t0, NULL);
    for (int i = 0; i < thread_count; i++)
        pthread_create(&threads[i], NULL, pool_worker, NULL);
    for (int i = 0; i < thread_count; i++)
        pthread_join(threads[i], NULL);
    gettimeofday(&t1, NULL);

    free(threads);
    free_tree(&bst_root);
    bst_root = NULL;

    return elapsed_ms(&t0, &t1);
}

/* ══════════════════════════════════════════════════════════════════════
 * §10 main
 * ══════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    srand(0);

    /* Generate D exactly once (uses rand() stream starting at seed 0) */
    int  D_size;
    Op  *D = generate_D(&D_size);
    printf("Generated %d operations.\n\n", D_size);

    /* ── LBST benchmark ── */
    double lbst_ms = bench_lbst(D, D_size);
    printf("LBST time          : %.2f ms\n\n", lbst_ms);

    /* ── pBST benchmark for MAX_THREAD_COUNT = 1 .. 128 ── */
    printf("%-14s  %s\n", "Thread count", "Time (ms)");
    printf("%-14s  %s\n", "------------", "---------");

    for (int tc = 1; tc <= 16; tc++) {
        double pbst_ms = bench_pbst(D, D_size, tc);
        printf("%-14d  %.2f\n", tc, pbst_ms);
        fflush(stdout);
    }

    free(D);
    return 0;
}