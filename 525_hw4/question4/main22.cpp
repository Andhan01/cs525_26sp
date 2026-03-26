 /*
 * Compile: mpic++ main22.cpp -o ./run2 -O2 -lm
 * Run    : mpirun -np <p> ./run <n>
 *   e.g.  mpirun -np 4 ./run2 2048
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

static inline double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static inline double A_value(int gi, int gj) {
    uint64_t v = (uint64_t)(gi + 1) * 1315423911ULL
               + (uint64_t)(gj + 1) * 2654435761ULL;
    return (double)(v % 1000ULL) / 1000.0;
}

static inline double x_value(int gj) {
    return (double)((gj % 1000) + 1) / 1000.0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 2048;
    if (argc > 1) n = atoi(argv[1]);

    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    int p_row = dims[0];
    int p_col = dims[1];

    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    if (cart_comm == MPI_COMM_NULL) {
        if (world_rank == 0) fprintf(stderr, "MPI_Cart_create failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    MPI_Comm row_comm, col_comm;
    int remain_row[2] = {0, 1};  /* 固定行，沿列通信 */
    int remain_col[2] = {1, 0};  /* 固定列，沿行通信 */
    MPI_Cart_sub(cart_comm, remain_row, &row_comm);
    MPI_Cart_sub(cart_comm, remain_col, &col_comm);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    int *row_sizes = (int*)malloc((size_t)p_row * sizeof(int));
    int *row_disp  = (int*)malloc((size_t)p_row * sizeof(int));
    int *col_sizes = (int*)malloc((size_t)p_col * sizeof(int));
    int *col_disp  = (int*)malloc((size_t)p_col * sizeof(int));

    if (!row_sizes || !row_disp || !col_sizes || !col_disp) {
        fprintf(stderr, "Rank %d: malloc failed for partition arrays.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int base_r = n / p_row, rem_r = n % p_row;
    int base_c = n / p_col, rem_c = n % p_col;

    for (int r = 0; r < p_row; r++)
        row_sizes[r] = (r < rem_r) ? (base_r + 1) : base_r;
    row_disp[0] = 0;
    for (int r = 1; r < p_row; r++)
        row_disp[r] = row_disp[r - 1] + row_sizes[r - 1];

    for (int c = 0; c < p_col; c++)
        col_sizes[c] = (c < rem_c) ? (base_c + 1) : base_c;
    col_disp[0] = 0;
    for (int c = 1; c < p_col; c++)
        col_disp[c] = col_disp[c - 1] + col_sizes[c - 1];

    int n_local_rows = row_sizes[my_row];
    int n_local_cols = col_sizes[my_col];

    double *A_block   = (double*)malloc((size_t)n_local_rows * (size_t)n_local_cols * sizeof(double));
    double *x_seg     = (double*)malloc((size_t)n_local_cols * sizeof(double));
    double *y_partial = (double*)malloc((size_t)n_local_rows * sizeof(double));
    double *y_row     = (double*)malloc((size_t)n_local_rows * sizeof(double));

    if ((!A_block && n_local_rows * n_local_cols > 0) ||
        (!x_seg && n_local_cols > 0) ||
        (!y_partial && n_local_rows > 0) ||
        (!y_row && n_local_rows > 0)) {
        fprintf(stderr, "Rank %d: malloc failed for local buffers.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* 本地矩阵块：用全局坐标确定值，保证整个 A 是一致的 */
    for (int i = 0; i < n_local_rows; i++) {
        int gi = row_disp[my_row] + i;
        for (int j = 0; j < n_local_cols; j++) {
            int gj = col_disp[my_col] + j;
            A_block[(size_t)i * (size_t)n_local_cols + (size_t)j] = A_value(gi, gj);
        }
    }

    /* x 的每个列分块由该列最上面（my_row == 0）的进程生成 */
    if (my_row == 0) {
        for (int j = 0; j < n_local_cols; j++) {
            int gj = col_disp[my_col] + j;
            x_seg[j] = x_value(gj);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now_us();

    /* 第一步：列内广播，让同一列的所有进程都拿到该列对应的 x 分块 */
    MPI_Bcast(x_seg, n_local_cols, MPI_DOUBLE, 0, col_comm);

    /* 第二步：局部矩阵-向量乘法 */
    for (int i = 0; i < n_local_rows; i++) {
        double sum = 0.0;
        const double *row = A_block + (size_t)i * (size_t)n_local_cols;
        for (int j = 0; j < n_local_cols; j++) {
            sum += row[j] * x_seg[j];
        }
        y_partial[i] = sum;
    }

    /* 第三步：行内归约，把同一行各列的 partial sum 加起来 */
    MPI_Reduce(y_partial, y_row, n_local_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    /* 只让“行归约根节点”加入收集最终结果的 communicator */
    MPI_Comm rowroot_comm = MPI_COMM_NULL;
    int is_rowroot = (row_rank == 0);
    MPI_Comm_split(cart_comm, is_rowroot ? 1 : MPI_UNDEFINED, my_row, &rowroot_comm);

    double *y_full = NULL;
    if (is_rowroot && my_row == 0) {
        y_full = (double*)malloc((size_t)n * sizeof(double));
        if (!y_full && n > 0) {
            fprintf(stderr, "Rank %d: malloc failed for y_full.\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (is_rowroot) {
        MPI_Gatherv(y_row, n_local_rows, MPI_DOUBLE,
                    y_full, row_sizes, row_disp, MPI_DOUBLE,
                    0, rowroot_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        double elapsed = now_us() - t0;
        printf("2D matvec: n=%d p=%d grid=%dx%d time=%.2f us\n",
               n, world_size, p_row, p_col, elapsed);

        if (y_full && n > 0) {
            printf("y[0]=%.6f, y[n-1]=%.6f\n", y_full[0], y_full[n - 1]);
        }
    }

    free(A_block);
    free(x_seg);
    free(y_partial);
    free(y_row);
    free(row_sizes);
    free(row_disp);
    free(col_sizes);
    free(col_disp);
    if (y_full) free(y_full);

    if (rowroot_comm != MPI_COMM_NULL) MPI_Comm_free(&rowroot_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}