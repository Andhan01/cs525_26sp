#define _GNU_SOURCE
/*
 * CS 52500 Spring 2026 — Homework #3, Question 2
 * Parallel Binary Search Tree (pBST)
 *
 * Q1 的 starter code 函数（randint / create_vertex / free_tree /
 * inorder_traversal_tree / print_tree）原封不动地沿用，仅在节点
 * 结构体中增加 pthread_rwlock_t，并在 create_vertex / free_tree
 * 中对应初始化 / 销毁该锁。
 */

#include <assert.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_THREAD_COUNT 2

/* ══════════════════════════════════════════════════════════════════════
 * Node  （在 Q1 结构体基础上增加 pthread_rwlock_t）
 * ══════════════════════════════════════════════════════════════════════ */
typedef struct vertex {
    int value;
    int deleted;          /* 保留字段，与 Q1 兼容；pBST 本身不使用 */
    struct vertex *left;
    struct vertex *right;
    pthread_rwlock_t lock;   /* 细粒度读写锁，Q2 新增 */
} vertex_t;

/* ══════════════════════════════════════════════════════════════════════
 * Starter-code helpers  ← 与 Q1 完全相同，一字不改
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
    vet->value   = value;
    vet->deleted = 0;
    vet->left    = NULL;
    vet->right   = NULL;
    pthread_rwlock_init(&vet->lock, NULL);   /* Q2 额外初始化锁 */
    return vet;
}

/*
 * free_tree: free the tree; 签名与 starter code 完全相同 (vertex_t **)
 */
void free_tree(vertex_t **root) {
    if (root == NULL || *root == NULL)
        return;
    free_tree(&(*root)->left);
    free_tree(&(*root)->right);
    pthread_rwlock_destroy(&(*root)->lock);  /* Q2 额外销毁锁 */
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
 * Global state
 * ══════════════════════════════════════════════════════════════════════ */
static vertex_t        *bst_root   = NULL;
static pthread_rwlock_t root_rwlock = PTHREAD_RWLOCK_INITIALIZER;
static sem_t            thread_sem;

/* ══════════════════════════════════════════════════════════════════════
 * pBST_insert — hand-over-hand write-locking
 *
 * 锁顺序（自顶向下，杜绝死锁）：
 *   root_rwlock(W) → curr(W) → unlock root_rwlock
 *                 → next(W)  → unlock curr → ...
 * ══════════════════════════════════════════════════════════════════════ */
void pBST_insert(int val) {

    pthread_rwlock_wrlock(&root_rwlock);

    /* 空树：直接插入根节点 */
    if (bst_root == NULL) {
        bst_root = create_vertex(val);
        pthread_rwlock_unlock(&root_rwlock);
        return;
    }

    vertex_t *curr = bst_root;
    pthread_rwlock_wrlock(&curr->lock);
    pthread_rwlock_unlock(&root_rwlock);   /* 持有 curr 后即可释放根锁 */

    while (1) {
        int       go_left = (curr->value >= val);
        vertex_t **slot   = go_left ? &curr->left : &curr->right;
        vertex_t  *next   = *slot;

        if (next == NULL) {               /* 找到插入位置 */
            *slot = create_vertex(val);
            pthread_rwlock_unlock(&curr->lock);
            return;
        }

        /* 手拉手：先锁子节点，再释放父节点 */
        pthread_rwlock_wrlock(&next->lock);
        pthread_rwlock_unlock(&curr->lock);
        curr = next;
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * pBST_delete — 标准 BST 物理删除，hand-over-hand write-locking
 *
 * 维持自顶向下的锁顺序：
 *   parent_lock → curr->lock，始终先锁子节点再释放父节点
 *
 * 双子节点情况：用独立的手拉手链找 inorder successor，
 * 全程保持 curr->lock 持有，不会破坏锁序。
 * ══════════════════════════════════════════════════════════════════════ */
void pBST_delete(int val) {

    pthread_rwlock_wrlock(&root_rwlock);

    if (!bst_root) {
        pthread_rwlock_unlock(&root_rwlock);
        return;
    }

    /* 用 parent_lock / child_ptr 抽象"根指针锁"和"子指针槽" */
    pthread_rwlock_t  *parent_lock = &root_rwlock;
    vertex_t         **child_ptr   = &bst_root;
    vertex_t          *curr        = bst_root;
    pthread_rwlock_wrlock(&curr->lock);

    /* ── 查找目标节点 ── */
    while (curr->value != val) {
        vertex_t **next_slot = (curr->value > val)
                                 ? &curr->left
                                 : &curr->right;
        vertex_t  *next      = *next_slot;

        if (!next) {                       /* 目标不存在，no-op */
            pthread_rwlock_unlock(parent_lock);
            pthread_rwlock_unlock(&curr->lock);
            return;
        }

        pthread_rwlock_wrlock(&next->lock);
        pthread_rwlock_unlock(parent_lock);
        parent_lock = &curr->lock;
        child_ptr   = next_slot;
        curr        = next;
    }

    /* ── 删除 curr ── */

    if (!curr->left && !curr->right) {
        /* 叶节点 */
        *child_ptr = NULL;
        pthread_rwlock_unlock(parent_lock);
        pthread_rwlock_unlock(&curr->lock);
        pthread_rwlock_destroy(&curr->lock);
        free(curr);

    } else if (!curr->left) {
        /* 只有右孩子 */
        *child_ptr  = curr->right;
        curr->right = NULL;
        pthread_rwlock_unlock(parent_lock);
        pthread_rwlock_unlock(&curr->lock);
        pthread_rwlock_destroy(&curr->lock);
        free(curr);

    } else if (!curr->right) {
        /* 只有左孩子 */
        *child_ptr = curr->left;
        curr->left = NULL;
        pthread_rwlock_unlock(parent_lock);
        pthread_rwlock_unlock(&curr->lock);
        pthread_rwlock_destroy(&curr->lock);
        free(curr);

    } else {
        /*
         * 双子节点：找 inorder successor（curr->right 子树中最左节点）。
         * succ_par->lock 持有顺序同样自顶向下，不会死锁。
         */
        vertex_t *succ_par = curr;         /* curr->lock 已持有   */
        vertex_t *succ     = curr->right;
        pthread_rwlock_wrlock(&succ->lock);

        while (succ->left) {
            vertex_t *next = succ->left;
            pthread_rwlock_wrlock(&next->lock);
            if (succ_par != curr)
                pthread_rwlock_unlock(&succ_par->lock);
            succ_par = succ;
            succ     = next;
        }

        /* 将后继值复制到 curr，物理删除 succ */
        curr->value = succ->value;

        if (succ_par == curr)
            curr->right    = succ->right;
        else {
            succ_par->left = succ->right;
            pthread_rwlock_unlock(&succ_par->lock);
        }

        pthread_rwlock_unlock(&succ->lock);
        succ->left = succ->right = NULL;
        pthread_rwlock_destroy(&succ->lock);
        free(succ);

        pthread_rwlock_unlock(parent_lock);
        pthread_rwlock_unlock(&curr->lock);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * pBST_lookup — hand-over-hand read-locking
 *
 * 读锁允许多个 lookup 并发执行，互不阻塞。
 * ══════════════════════════════════════════════════════════════════════ */
bool pBST_lookup(int val) {

    pthread_rwlock_rdlock(&root_rwlock);
    vertex_t *curr = bst_root;

    if (!curr) {
        pthread_rwlock_unlock(&root_rwlock);
        return false;
    }

    pthread_rwlock_rdlock(&curr->lock);
    pthread_rwlock_unlock(&root_rwlock);

    while (curr) {
        if (curr->value == val) {
            pthread_rwlock_unlock(&curr->lock);
            return true;
        }

        vertex_t *next = (curr->value > val)
                           ? curr->left
                           : curr->right;

        if (!next) {
            pthread_rwlock_unlock(&curr->lock);
            return false;
        }

        pthread_rwlock_rdlock(&next->lock);
        pthread_rwlock_unlock(&curr->lock);
        curr = next;
    }

    return false;
}

/* ══════════════════════════════════════════════════════════════════════
 * 线程调度
 *
 * sem_t thread_sem（初值 = MAX_THREAD_COUNT = 2）：
 *   每次 dispatch 前 sem_wait，线程结束后 sem_post，
 *   保证同时运行的操作线程 ≤ MAX_THREAD_COUNT。
 * ══════════════════════════════════════════════════════════════════════ */
typedef struct {
    int  type;       /* 0 = insert, 1 = delete, 2 = lookup */
    int  val;
    bool result;
    bool done;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} OpTask;

static void *run_op(void *arg) {
    OpTask *t = (OpTask *)arg;

    if      (t->type == 0) pBST_insert(t->val);
    else if (t->type == 1) pBST_delete(t->val);
    else                   t->result = pBST_lookup(t->val);

    pthread_mutex_lock(&t->mutex);
    t->done = true;
    pthread_cond_signal(&t->cond);
    pthread_mutex_unlock(&t->mutex);

    sem_post(&thread_sem);
    return NULL;
}

static bool dispatch(int type, int val) {
    sem_wait(&thread_sem);          /* 等待可用线程槽 */

    OpTask *task = (OpTask *)malloc(sizeof(OpTask));
    assert(task);
    task->type   = type;
    task->val    = val;
    task->result = false;
    task->done   = false;
    pthread_mutex_init(&task->mutex, NULL);
    pthread_cond_init(&task->cond,   NULL);

    pthread_t t;
    pthread_create(&t, NULL, run_op, task);
    pthread_detach(t);

    /* 等待本次任务完成后再返回（保证顺序输出） */
    pthread_mutex_lock(&task->mutex);
    while (!task->done)
        pthread_cond_wait(&task->cond, &task->mutex);
    bool res = task->result;
    pthread_mutex_unlock(&task->mutex);

    pthread_mutex_destroy(&task->mutex);
    pthread_cond_destroy(&task->cond);
    free(task);
    return res;
}

/* ══════════════════════════════════════════════════════════════════════
 * main
 * ══════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    srand(0);
    sem_init(&thread_sem, 0, MAX_THREAD_COUNT);

    /* ── Part 1: Parallel Insert (operations 22–31) ── */
    dispatch(0,  5);   /* (22) */
    dispatch(0,  2);   /* (23) */
    dispatch(0,  8);   /* (24) */
    dispatch(0,  1);   /* (25) */
    dispatch(0,  9);   /* (26) */
    dispatch(0,  3);   /* (27) */
    dispatch(0,  6);   /* (28) */
    dispatch(0,  7);   /* (29) */
    dispatch(0,  4);   /* (30) */
    dispatch(0, 10);   /* (31) */

    printf("=== Part 1: Final pBST after Parallel Insert ===\n");
    print_tree(bst_root, 0);
    printf("Inorder: ");
    inorder_traversal_tree(bst_root);
    printf("\n\n");

    /* ── Part 2: Parallel Delete (operations 32–36) ── */
    dispatch(1,  7);   /* (32) */
    dispatch(1,  0);   /* (33) — not in tree, no-op */
    dispatch(1,  6);   /* (34) */
    dispatch(1,  2);   /* (35) */
    dispatch(1, 10);   /* (36) */

    printf("=== Part 2: Final pBST after Parallel Delete ===\n");
    print_tree(bst_root, 0);
    printf("Inorder: ");
    inorder_traversal_tree(bst_root);
    printf("\n\n");

    /* ── Part 3: Parallel Look-up (operations 37–42) ── */
    printf("=== Part 3: Parallel Look-up ===\n");
    printf("look_up  1 (37): %s\n", dispatch(2,  1) ? "true" : "false");
    printf("look_up  6 (38): %s\n", dispatch(2,  6) ? "true" : "false");
    printf("look_up 10 (39): %s\n", dispatch(2, 10) ? "true" : "false");
    printf("look_up  5 (40): %s\n", dispatch(2,  5) ? "true" : "false");
    printf("look_up  8 (41): %s\n", dispatch(2,  8) ? "true" : "false");
    printf("look_up  0 (42): %s\n", dispatch(2,  0) ? "true" : "false");

    free_tree(&bst_root);
    sem_destroy(&thread_sem);
    return 0;
}