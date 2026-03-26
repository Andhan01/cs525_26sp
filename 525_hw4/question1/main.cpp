/*
 * CS 525 HW4 – Question 1
 * Ping-pong messages of increasing lengths (1K … 100K bytes).
 * Compute RTT, divide by 2 × NUM_ROUNDS to get one-way communication time.
 *
 * Compile: mpic++ main.cpp -o ./run
 * Run    : mpirun -np 2 ./run
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Number of ping-pong rounds per message size */
#define NUM_ROUNDS 100

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Need at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    /* Message sizes: 1 K … 100 K bytes */
    int msg_sizes[] = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 102400
    };
    int num_sizes = (int)(sizeof(msg_sizes) / sizeof(msg_sizes[0]));

    if (rank == 0) {
        printf("%-20s  %-20s\n", "MsgSize(bytes)", "CommTime(us)");
        printf("%-20s  %-20s\n", "--------------", "------------");
    }

    for (int s = 0; s < num_sizes; s++) {
        int  msg_size = msg_sizes[s];
        char *buf     = (char *)malloc(msg_size);
        memset(buf, 1, msg_size);

        struct timeval t0, t1;
        double elapsed = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            gettimeofday(&t0, NULL);
            for (int r = 0; r < NUM_ROUNDS; r++) {
// buf: Initial address of send buffer (choice).
// count: Number of elements in send buffer (nonnegative integer).
// datatype: Datatype of each send buffer element (handle).
// dest: Rank of destination (integer).
// tag: Message tag (integer).
// comm: Communicator (handle).
                MPI_Send(buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
            gettimeofday(&t1, NULL);

            /* RTT / (2 * rounds) = one-way communication time */
            double rtt_us = (t1.tv_sec  - t0.tv_sec ) * 1e6 +
                            (t1.tv_usec - t0.tv_usec);
            elapsed = rtt_us / (2.0 * NUM_ROUNDS);

            printf("%-20d  %-20.4f\n", msg_size, elapsed);

        } else if (rank == 1) {
            for (int r = 0; r < NUM_ROUNDS; r++) {
                MPI_Recv(buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        free(buf);
    }

    MPI_Finalize();
    return 0;
}