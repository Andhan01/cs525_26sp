#include <stdio.h>
#include <mpi.h>



int mpi_quick_sort(int * queue){
    
}



int main(int argc, char** argv){
    // int process_Rank, size_Of_Cluster;

    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    // MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

    // printf("Hello World from process %d of %d\n", process_Rank, size_Of_Cluster);

    // MPI_Finalize();
    // return 0;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int gsize,*sendbuf, process_Rank;
    int root=0;
    int rbuf[100];
    // ...
    MPI_Comm_size(comm, &gsize);
    MPI_Comm_rank(comm, &process_Rank);
    if (process_Rank == root) {
        sendbuf = (int *)malloc(gsize * 100 * sizeof(int));
        for (int i = 0; i < gsize * 100; i++) {
            sendbuf[i] = i;  // 初始化数据
        }
    }
// int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
//      void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
//      MPI_Comm comm)
    MPI_Scatter(sendbuf, 100, MPI_INT, rbuf, 100, MPI_INT, root, comm);
    MPI_Finalize();
}


// mpiicc hello_world_mpi.cpp -o hello_world_mpi.exe
// mpiicc main.c -o main.exe

// mpic++ main.c -o main.exe


//mpirun -np 4 ./main.exe