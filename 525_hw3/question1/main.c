/*
 * CS 52500 Spring 2026 - Homework #3, Question 1
 * Lazy Binary Search Tree (LBST)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* ══════════════════════════════════════════════════════════════════════
 * Node definition
 * Extra fields size / n_marked are maintained internally to support
 * the *-BST check (Definition 1) and the half-marked check in O(1).
 * ══════════════════════════════════════════════════════════════════════ */
typedef struct vertex {
    int value;
    int deleted;          /* 0 = live, 1 = logically deleted (marked)   */
    struct vertex *left;
    struct vertex *right;
    int size;             /* total nodes in this subtree (incl. marked)  */
    int n_marked;         /* number of marked nodes in this subtree      */
} vertex_t;

/* ══════════════════════════════════════════════════════════════════════
 * Starter-code helpers  (provided verbatim / extended as needed)
 * ══════════════════════════════════════════════════════════════════════ */

/*
 * randint: generate an integer in range [lower, upper] uniformly at random
 */
int randint(int lower, int upper) {
    return (lower + rand() % (upper - lower));
}

vertex_t *create_vertex(int value) {
    vertex_t *vet = (vertex_t *)malloc(sizeof(vertex_t));
    assert(vet != NULL);
    vet->value    = value;
    vet->deleted  = 0;
    vet->left     = NULL;
    vet->right    = NULL;
    vet->size     = 1;
    vet->n_marked = 0;
    return vet;
}

/*
 * free_tree: free the entire tree and set *root to NULL.
 * Signature matches the starter code exactly (vertex_t **).
 */
void free_tree(vertex_t **root) {
    if (root == NULL || *root == NULL)
        return;
    free_tree(&(*root)->left);
    free_tree(&(*root)->right);
    free(*root);
    *root = NULL;
}

/*
 * inorder_traversal_tree: in-order traversal, skipping marked nodes.
 */
void inorder_traversal_tree(vertex_t *root) {
    if (root == NULL)
        return;
    inorder_traversal_tree(root->left);
    if (root->deleted != 1)
        printf("%d ", root->value);
    inorder_traversal_tree(root->right);
}

/*
 * print_tree: print an indented diagram of the tree.
 * Marked nodes are annotated with "(deleted)".
 */
void print_tree(vertex_t *root, int level) {
    if (root == NULL)
        return;
    printf("|");
    for (int i = 0; i < level; i++)
        printf("-");
    printf(" %d", root->value);
    if (root->deleted == 1)
        printf(" (deleted)");
    printf("\n");
    print_tree(root->left,  level + 1);
    print_tree(root->right, level + 1);
}

/* ══════════════════════════════════════════════════════════════════════
 * Internal bookkeeping
 * ══════════════════════════════════════════════════════════════════════ */

static int sz(vertex_t *v)  { return v ? v->size     : 0; }
static int nm(vertex_t *v)  { return v ? v->n_marked : 0; }

static void update(vertex_t *v) {
    if (!v) return;
    v->size     = 1 + sz(v->left) + sz(v->right);
    v->n_marked = v->deleted + nm(v->left) + nm(v->right);
}

/* ══════════════════════════════════════════════════════════════════════
 * Dynamic array  (replaces std::vector for pure C)
 * ══════════════════════════════════════════════════════════════════════ */
typedef struct {
    vertex_t **data;
    int        size;
    int        capacity;
} node_array_t;

static void array_init(node_array_t *a) {
    a->size     = 0;
    a->capacity = 16;
    a->data     = malloc(a->capacity * sizeof(vertex_t *));
    assert(a->data);
}

static void array_push(node_array_t *a, vertex_t *v) {
    if (a->size >= a->capacity) {
        a->capacity *= 2;
        a->data = realloc(a->data, a->capacity * sizeof(vertex_t *));
        assert(a->data);
    }
    a->data[a->size++] = v;
}

static void array_free_struct(node_array_t *a) {
    free(a->data);
}

/* ══════════════════════════════════════════════════════════════════════
 * Rebuild helpers
 * ══════════════════════════════════════════════════════════════════════ */

/*
 * collect_inorder: visit every node in BST order.
 *   skip_marked = 0  → collect all  (used by rebalance_insert)
 *   skip_marked = 1  → skip marked  (used by rebalance_delete)
 */
static void collect_inorder(vertex_t *root, node_array_t *out, int skip_marked) {
    if (!root) return;
    collect_inorder(root->left,  out, skip_marked);
    if (!skip_marked || !root->deleted)
        array_push(out, root);
    collect_inorder(root->right, out, skip_marked);
}

/*
 * build_balanced: given nodes[l..r] already in sorted order,
 * wire them into a height-balanced BST via the midpoint rule.
 */
static vertex_t *build_balanced(vertex_t **nodes, int l, int r) {
    if (l > r) return NULL;
    int mid        = (l + r) / 2;
    vertex_t *root = nodes[mid];
    root->left     = build_balanced(nodes, l,       mid - 1);
    root->right    = build_balanced(nodes, mid + 1, r);
    update(root);
    return root;
}

/*
 * rebalance_insert: rebuild keeping ALL nodes (including marked).
 * Called when the subtree violates the *-BST condition.
 */
static vertex_t *rebalance_insert(vertex_t *vr) {
    node_array_t arr;
    array_init(&arr);
    collect_inorder(vr, &arr, 0);          /* collect all        */

    /* disconnect every node before rewiring */
    for (int i = 0; i < arr.size; i++)
        arr.data[i]->left = arr.data[i]->right = NULL;

    vertex_t *new_root = build_balanced(arr.data, 0, arr.size - 1);
    array_free_struct(&arr);
    return new_root;
}

/*
 * rebalance_delete: rebuild keeping only LIVE nodes;
 * marked nodes are physically freed.
 * Called when more than half the subtree is marked.
 */
static vertex_t *rebalance_delete(vertex_t *vr) {
    node_array_t all, live;
    array_init(&all);
    array_init(&live);

    collect_inorder(vr, &all, 0);          /* collect all        */

    for (int i = 0; i < all.size; i++) {
        vertex_t *n = all.data[i];
        n->left = n->right = NULL;         /* disconnect first   */
        if (!n->deleted)
            array_push(&live, n);
        else
            free(n);                        /* physically remove  */
    }

    vertex_t *new_root = NULL;
    if (live.size > 0)
        new_root = build_balanced(live.data, 0, live.size - 1);

    array_free_struct(&all);
    array_free_struct(&live);
    return new_root;
}

/* ══════════════════════════════════════════════════════════════════════
 * Algorithm 1 — Lazy Insert
 *
 * Condition (*-BST, Definition 1):
 *   ||TL| - |TR|| >= (1/3)|T|  ⟺  3 * |left - right| >= size
 * ══════════════════════════════════════════════════════════════════════ */
vertex_t *lazy_insert(vertex_t *vr, int val) {
    /* Base case */
    if (vr == NULL)
        return create_vertex(val);

    /* Recurse — equal values go to the left subtree (Algorithm 1, line 6) */
    if (vr->value >= val)
        vr->left  = lazy_insert(vr->left,  val);
    else
        vr->right = lazy_insert(vr->right, val);

    update(vr);

    /* *-BST check (Definition 1) */
    int diff = sz(vr->left) - sz(vr->right);
    if (diff < 0) diff = -diff;
    if (3 * diff >= vr->size)
        vr = rebalance_insert(vr);

    return vr;
}

/* ══════════════════════════════════════════════════════════════════════
 * Algorithm 2 — Lazy Delete
 *
 * Nodes are MARKED, not removed.  Physical removal happens only during
 * rebalance_delete, triggered when marked > |T| / 2.
 *
 * Note on Algorithm 2 line 6:
 *   The pseudocode shows ">=" but the "else" (mark) branch would then be
 *   unreachable.  The intended semantics — confirmed by the comment
 *   "Try to find the value" — require strict inequality:
 *     > val  → go left
 *     < val  → go right
 *     == val → mark
 * ══════════════════════════════════════════════════════════════════════ */
vertex_t *lazy_delete(vertex_t *vr, int val) {
    /* Element not found — nothing to do */
    if (vr == NULL)
        return NULL;

    if (vr->value > val)
        vr->left  = lazy_delete(vr->left,  val);
    else if (vr->value < val)
        vr->right = lazy_delete(vr->right, val);
    else
        /* Found: mark instead of physically removing */
        vr->deleted = 1;

    update(vr);

    /* Rebuild when more than half the nodes are marked */
    if (vr->n_marked * 2 > vr->size)
        vr = rebalance_delete(vr);

    return vr;
}

/* ══════════════════════════════════════════════════════════════════════
 * Look-up  (Note 4: same algorithm as a regular sequential BST)
 *
 * Returns 1 (true)  if the value exists AND is not marked.
 * Returns 0 (false) otherwise.
 * ══════════════════════════════════════════════════════════════════════ */
int look_up(vertex_t *vr, int val) {
    if (vr == NULL)
        return 0;
    if (vr->value == val)
        return !vr->deleted;
    if (vr->value > val)
        return look_up(vr->left,  val);
    return look_up(vr->right, val);
}

/* ══════════════════════════════════════════════════════════════════════
 * main
 * ══════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    srand(0);

    vertex_t *root = NULL;

    /* ── Part 1: Lazy Insert (operations 1–10) ── */
    root = lazy_insert(root, 8);   /* (1)  */
    root = lazy_insert(root, 7);   /* (2)  */
    root = lazy_insert(root, 9);   /* (3)  */
    root = lazy_insert(root, 1);   /* (4)  */
    root = lazy_insert(root, 5);   /* (5)  */
    root = lazy_insert(root, 2);   /* (6)  */
    root = lazy_insert(root, 6);   /* (7)  */
    root = lazy_insert(root, 3);   /* (8)  */
    root = lazy_insert(root, 4);   /* (9)  */
    root = lazy_insert(root, 0);   /* (10) */

    printf("=== Part 1: Final LBST after Lazy Insert ===\n");
    print_tree(root, 0);
    printf("Inorder (live nodes): ");
    inorder_traversal_tree(root);
    printf("\n\n");

    /* ── Part 2: Lazy Delete (operations 11–15) ── */
    root = lazy_delete(root,  7);  /* (11) */
    root = lazy_delete(root,  0);  /* (12) */
    root = lazy_delete(root,  6);  /* (13) */
    root = lazy_delete(root,  2);  /* (14) */
    root = lazy_delete(root, 10);  /* (15) — not in tree, no-op */

    printf("=== Part 2: Final LBST after Lazy Delete (highlighted marked nodes) ===\n");
    print_tree(root, 0);
    printf("Inorder (live nodes): ");
    inorder_traversal_tree(root);
    printf("\n\n");

    /* ── Part 3: Look-up (operations 16–21) ── */
    printf("=== Part 3: Look-up ===\n");
    printf("look_up  1 (16): %d\n", look_up(root,  1));
    printf("look_up  6 (17): %d\n", look_up(root,  6));
    printf("look_up 10 (18): %d\n", look_up(root, 10));
    printf("look_up  5 (19): %d\n", look_up(root,  5));
    printf("look_up  8 (20): %d\n", look_up(root,  8));
    printf("look_up  9 (21): %d\n", look_up(root,  9));

    free_tree(&root);
    return 0;
}