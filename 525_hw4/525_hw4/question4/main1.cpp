/*
 * CS 525 HW4 – Question 4  (Row-wise 1-D partitioning)
 * Matrix-Vector Multiplication: y = A * x
 *
 * Partitioning:
 *   Each processor owns n_local = n/p consecutive rows of A and the
 *   corresponding n_local elements of y.  x is replicated via MPI_Allgather.
 *
 * Timing (per HW spec):
 *   1. MPI_Barrier to synchronise all procs.
 *   2. Start clock on proc 0.
 *   3. Allgather x, then compute local mat-vec product.
 *   4. MPI_Barrier.
 *   5. Stop clock on proc 0.
 *
 * Compile: mpic++ main1.cpp -o ./run1 -O2
 * Run    : mpirun -np <p> ./run1 <n>
 *   e.g.  mpirun -np 4 ./run1 2048
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static inline double now_us()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 2048;
    if (argc > 1) n = atoi(argv[1]);

    /* -------------------------------
       Non-uniform row partition
       ------------------------------- */
    int base = n / size;
    int rem  = n % size;

    int n_local = (rank < rem) ? base + 1 : base;

    /* -------------------------------
       Allocate memory
       ------------------------------- */
    double *A_local = (double *)malloc((size_t)n_local * n * sizeof(double));
    double *x_local = (double *)malloc((size_t)n_local * sizeof(double));
    double *x_full  = (double *)malloc((size_t)n       * sizeof(double));
    double *y_local = (double *)malloc((size_t)n_local * sizeof(double));

    if (!A_local || !x_local || !x_full || !y_local) {
        fprintf(stderr, "Rank %d: malloc failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* -------------------------------
       Build counts and displacements
       ------------------------------- */
    int *counts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        counts[i] = (i < rem) ? base + 1 : base;
    }

    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + counts[i-1];
    }

    /* -------------------------------
       Generate data independently
       ------------------------------- */
    srand((unsigned)(rank * 9876 + 1));

    for (int i = 0; i < n_local * n; i++)
        A_local[i] = (double)rand() / RAND_MAX;

    for (int i = 0; i < n_local; i++)
        x_local[i] = (double)rand() / RAND_MAX;

    /* -------------------------------
       Timing start
       ------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = (rank == 0) ? now_us() : 0.0;

    /* -------------------------------
       Step 1: Allgatherv (critical fix)
       ------------------------------- */
    MPI_Allgatherv(x_local, n_local, MPI_DOUBLE,
                   x_full, counts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    /* -------------------------------
       Step 2: Local computation
       ------------------------------- */
    for (int i = 0; i < n_local; i++) {
        double sum = 0.0;
        const double *row = A_local + (size_t)i * n;

        for (int j = 0; j < n; j++)
            sum += row[j] * x_full[j];

        y_local[i] = sum;
    }
    /* -------------------------------
       Step 3: 汇总
       ------------------------------- */
    double *y_full = NULL;
    if (rank == 0)
        y_full = (double *)malloc(n * sizeof(double));

    MPI_Gatherv(y_local, n_local, MPI_DOUBLE,
                y_full, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    /* -------------------------------
       Timing end
       ------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = now_us() - t_start;
        printf("1D n=%d p=%d time=%.2f us\n", n, size, elapsed);
    }

    /* -------------------------------
       Free memory
       ------------------------------- */
    free(A_local);
    free(x_local);
    free(x_full);
    free(y_local);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
