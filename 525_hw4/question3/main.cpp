/*
 * CS 525 HW4 – Question 3
 * Message-passing Quicksort (slide 60, sorting lecture).
 *
 * Algorithm (hypercube / communicator-splitting variant):
 *   while communicator size > 1:
 *     1. Rank 0 in current communicator selects and broadcasts a pivot.
 *     2. Each process partitions its local data into low (≤ pivot) and
 *        high (> pivot) halves.
 *     3. Low-ranked half (rank < half) keeps low data; swaps high data
 *        with its partner in the upper half.
 *     4. High-ranked half keeps high data; swaps low data with partner.
 *     5. Split communicator: lower procs → color 0, upper → color 1.
 *   After the loop each process sorts its local data sequentially.
 *
 * Compile: mpic++ main.cpp -o ./run -std=c++11
 * Run    : mpirun -np <p> ./run <N>
 *   e.g.  mpirun -np 8 ./run 1000000
 *
 * p must be a power of two.  N = total element count (default 100000).
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <vector>

/* ------------------------------------------------------------------ */
static inline double now_us()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 100000LL;
    if (argc > 1) N = atoll(argv[1]);

    /* Distribute elements evenly; last rank absorbs remainder. */
    long long base    = N / size;
    long long rem     = N % size;
    long long local_n = base + (rank == size - 1 ? rem : 0);

    std::vector<int> data(local_n);
    srand((unsigned)(rank * 1234567 + 89));
    for (long long i = 0; i < local_n; i++)
        data[i] = rand();

    /* ---- barrier, then start timing on rank 0 ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = (rank == 0) ? now_us() : 0.0;

    /* ---- parallel quicksort ---- */
    MPI_Comm cur_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &cur_comm);   /* own handle we can free */

    int cur_rank = rank;
    int cur_size = size;

    while (cur_size > 1) {
        /* ---- Step 1: rank 0 in this comm selects pivot ---- */
        int pivot = 0;
        if (cur_rank == 0 && !data.empty())
            pivot = data[data.size() / 2];
        MPI_Bcast(&pivot, 1, MPI_INT, 0, cur_comm);//如果当前运行的处理器是
        // root跑到这就发送pivot，如果是非root跑到这就接受pivot
// buffer: Starting address of buffer (choice).
// count: Number of entries in buffer (integer).
// datatype: Data type of buffer (handle).
// root: Rank of broadcast root (integer).
// comm: Communicator (handle).
        /* ---- Step 2: partition local data ---- */
        std::vector<int> low_part, high_part;
        low_part.reserve(data.size());
        high_part.reserve(data.size());
        for (int x : data) {
            if (x <= pivot) low_part.push_back(x);
            else            high_part.push_back(x);
        }

        /* ---- Step 3/4: exchange with partner ---- */
        int half    = cur_size / 2;
        int color   = (cur_rank < half) ? 0 : 1;   /* 0=lower, 1=upper */

        if (cur_rank < half) {
            int partner = cur_rank + half;

            /* tell partner how many high elements we'll send */
            long long send_count = (long long)high_part.size();
            long long recv_count = 0;
            MPI_Sendrecv(&send_count, 1, MPI_LONG_LONG, partner, 10,
                         &recv_count, 1, MPI_LONG_LONG, partner, 10,
                         cur_comm, MPI_STATUS_IGNORE);
// sendbuf: Initial address of send buffer (choice).
// sendcount: Number of elements to send (integer).
// sendtype: Type of elements in send buffer (handle).
// dest: Rank of destination (integer).
// sendtag: Send tag (integer).
// recvcount: Maximum number of elements to receive (integer).
// recvtype: Type of elements in receive buffer (handle).
// source: Rank of source (integer).
// recvtag: Receive tag (integer).
// comm: Communicator (handle).
            std::vector<int> incoming(recv_count);
            MPI_Sendrecv(high_part.data(), (int)send_count, MPI_INT, partner, 11,
                         incoming.data(), (int)recv_count, MPI_INT, partner, 11,
                         cur_comm, MPI_STATUS_IGNORE);

            /* keep low + received-low from partner */
            data.clear();
            data.insert(data.end(), low_part.begin(),  low_part.end());
            data.insert(data.end(), incoming.begin(), incoming.end());

        } else {
            int partner = cur_rank - half;

            /* send low elements to partner, receive partner's high elements */
            long long send_count = (long long)low_part.size();
            long long recv_count = 0;
            MPI_Sendrecv(&send_count, 1, MPI_LONG_LONG, partner, 10,
                         &recv_count, 1, MPI_LONG_LONG, partner, 10,
                         cur_comm, MPI_STATUS_IGNORE);

            std::vector<int> incoming(recv_count);
            MPI_Sendrecv(low_part.data(), (int)send_count, MPI_INT, partner, 11,
                         incoming.data(), (int)recv_count, MPI_INT, partner, 11,
                         cur_comm, MPI_STATUS_IGNORE);

            /* keep high + received-high from partner */
            data.clear();
            data.insert(data.end(), high_part.begin(), high_part.end());
            data.insert(data.end(), incoming.begin(), incoming.end());
        }

        /* ---- Step 5: split communicator ---- */
        MPI_Comm new_comm;
        MPI_Comm_split(cur_comm, color, cur_rank, &new_comm);
        MPI_Comm_free(&cur_comm);
        cur_comm = new_comm;
        MPI_Comm_rank(cur_comm, &cur_rank);
        MPI_Comm_size(cur_comm, &cur_size);
    }

    /* ---- local sequential sort ---- */
    std::sort(data.begin(), data.end());

    /* ---- barrier, then stop timing on rank 0 ---- */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = now_us() - t_start;
        printf("N=%lld p=%d time=%.2f us\n", N, size, elapsed);
    }

    MPI_Comm_free(&cur_comm);
    MPI_Finalize();
    return 0;
}

//mpic++ main.cpp -o ./run -std=c++11