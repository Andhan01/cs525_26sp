/*
 * CS 525 HW4 – Question 4  (2-D partitioning)
 * Matrix-Vector Multiplication: y = A * x
 *
 * Processor grid  p_row × p_col  (chosen as square as possible).
 *   p_col = largest divisor of p that is ≤ sqrt(p).
 *   p_row = p / p_col.
 *   Processor (r, c) = rank  r * p_col + c.
 *
 * Data layout:
 *   A_block  : (n/p_row) × (n/p_col)  sub-matrix owned by proc (r,c)
 *   x_seg    : n/p_col elements – the c-th segment of vector x
 *              (generated independently, no communication of x needed)
 *
 * Algorithm (Page 11 of lecture slides):
 *   1. Each proc (r,c) computes  y_partial = A_block × x_seg
 *      (size n/p_row vector).
 *   2. Reduce y_partial within each processor row (row communicator)
 *      → proc (r, 0) accumulates y_r = Σ_c A_rc × x_c.
 *
 * Timing (per HW spec): Barrier → start clock (proc 0) →
 *   compute + communicate → Barrier → stop clock (proc 0).
 *
 * Compile: mpic++ main2.cpp -o ./run2 -O2 -lm
 * Run    : mpirun -np <p> ./run <n>
 *   e.g.  mpirun -np 4 ./run2 2048
 *
 * Note: n must be divisible by both p_row and p_col.
 *       For p = 1,2,4,8  the grid is  1×1, 1×2, 2×2, 2×4.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

static inline double now_us()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

/* Find near-square grid */
static void best_grid(int p, int *p_row, int *p_col)
{
    int q = (int)sqrt((double)p);
    while (q > 0 && p % q != 0) q--;
    *p_row = q;
    *p_col = p / q;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 2048;
    if (argc > 1) n = atoi(argv[1]);

    int p_row, p_col;
    best_grid(size, &p_row, &p_col);

    int my_row = rank / p_col;
    int my_col = rank % p_col;

    /* -------------------------------
       Non-uniform 2D partition
       ------------------------------- */
    int base_row = n / p_row;
    int rem_row  = n % p_row;
    int n_row = (my_row < rem_row) ? base_row + 1 : base_row;

    int base_col = n / p_col;
    int rem_col  = n % p_col;
    int n_col = (my_col < rem_col) ? base_col + 1 : base_col;

    /* -------------------------------
       Allocate memory
       ------------------------------- */
    double *A_block   = (double*) malloc((size_t)n_row * n_col * sizeof(double));
    double *x_seg     = (double*) malloc((size_t)n_col * sizeof(double));
    double *y_partial = (double*) malloc((size_t)n_row * sizeof(double));
    double *y_local   = (double*) malloc((size_t)n_row * sizeof(double));

    if (!A_block || !x_seg || !y_partial || !y_local) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* -------------------------------
       Generate data
       ------------------------------- */
    srand((unsigned)(rank * 13579 + 1));

    for (int i = 0; i < n_row * n_col; i++)
        A_block[i] = (double)rand() / RAND_MAX;

    for (int i = 0; i < n_col; i++)
        x_seg[i] = (double)rand() / RAND_MAX;

    /* -------------------------------
       Row communicator
       ------------------------------- */
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);

    /* -------------------------------
       Timing start
       ------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = (rank == 0) ? now_us() : 0.0;

    /* -------------------------------
       Step 1: local matvec
       ------------------------------- */
    for (int i = 0; i < n_row; i++) {
        double sum = 0.0;
        const double *row_ptr = A_block + (size_t)i * n_col;
        for (int j = 0; j < n_col; j++)
            sum += row_ptr[j] * x_seg[j];
        y_partial[i] = sum;
    }

    /* -------------------------------
       Step 2: reduce within row
       ------------------------------- */
    MPI_Reduce(y_partial, y_local, n_row,
               MPI_DOUBLE, MPI_SUM, 0, row_comm);

    /* -------------------------------
       Timing end
       ------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = now_us() - t_start;
        printf("2D n=%d p=%d grid=%dx%d time=%.2f us\n",
               n, size, p_row, p_col, elapsed);
    }

    /* -------------------------------
       Cleanup
       ------------------------------- */
    MPI_Comm_free(&row_comm);
    free(A_block);
    free(x_seg);
    free(y_partial);
    free(y_local);

    MPI_Finalize();
    return 0;
}