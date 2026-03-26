/*
 * CS 525 HW4 – Question 2
 * All processor pairs ping-pong simultaneously.
 * Pair low half (rank i) with high half (rank i + p/2).
 * Run with p = 2, 4, 8, 16, 32 to study congestion.
 *
 * Compile: mpic++ main.cpp -o ./run
 * Run    : mpirun -np <p> ./run
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_ROUNDS 100

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2 || size % 2 != 0) {
        if (rank == 0)
            fprintf(stderr, "Need an even number of processes (>= 2).\n");
        MPI_Finalize();
        return 1;
    }

    int half    = size / 2;
    int partner = (rank < half) ? rank + half : rank - half;
    int is_sender = (rank < half);   /* low half initiates */

    /* Message sizes: 1 K … 100 K bytes */
    int msg_sizes[] = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 102400
    };
    int num_sizes = (int)(sizeof(msg_sizes) / sizeof(msg_sizes[0]));

    if (rank == 0) {
        printf("p=%d\n", size);
        printf("%-20s  %-20s\n", "MsgSize(bytes)", "CommTime(us)");
        printf("%-20s  %-20s\n", "--------------", "------------");
    }

    for (int s = 0; s < num_sizes; s++) {
        int  msg_size = msg_sizes[s];
        char *buf     = (char *)malloc(msg_size);
        memset(buf, 1, msg_size);

        struct timeval t0, t1;
        double elapsed_local = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);

        if (is_sender) {
            gettimeofday(&t0, NULL);
            for (int r = 0; r < NUM_ROUNDS; r++) {
                MPI_Send(buf, msg_size, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, msg_size, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
            gettimeofday(&t1, NULL);

            double rtt_us = (t1.tv_sec  - t0.tv_sec ) * 1e6 +
                            (t1.tv_usec - t0.tv_usec);
            elapsed_local = rtt_us / (2.0 * NUM_ROUNDS);

        } else {
            /* high-half: receiver role */
            for (int r = 0; r < NUM_ROUNDS; r++) {
                MPI_Recv(buf, msg_size, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(buf, msg_size, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
            }
        }

        /*
         * Gather all sender timings to rank 0 and report the average.
         * Only ranks in the low half (senders) participate.
         */
        if (is_sender) {
            double avg_time = 0.0;
            MPI_Reduce(&elapsed_local, &avg_time, 1, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD);
// sendbuf: Address of send buffer (choice).
// receivebuf?
// count: Number of elements in send buffer (integer).
// datatype: Data type of elements of send buffer (handle).
// op: Reduce operation (handle).
// root: Rank of root process (integer).
// comm: Communicator (handle).
// info: Info (handle, persistent).
            if (rank == 0) {
                avg_time /= (double)half;
                printf("%-20d  %-20.4f\n", msg_size, avg_time);
            }
        } else {
            /* high-half still needs to participate in the Reduce */
            double dummy = 0.0;
            MPI_Reduce(&dummy, NULL, 1, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD);
        }

        free(buf);
    }

    MPI_Finalize();
    return 0;
}