#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
// #define N 500
#define BS 32

double multiply(int N);
double multiply_tile(int N);


int main() {
    float *rate1 = (float *)malloc(4 * sizeof(float));
    float *rate2 = (float *)malloc(4 * sizeof(float));
    int lst[4] = {100,500,1000,5000};
    printf("N, Rate_Standard, Rate_Tiled\n");
    for (int iter = 0; iter < 4; iter++){
        int N = lst[iter];
        rate1[iter] = 2.0*N*N*N/multiply(N);
        rate2[iter] = 2.0*N*N*N/multiply_tile(N);
        printf("%d, %f, %f\n", N, rate1[iter], rate2[iter]);
    }


    FILE *fp = fopen("record.csv", "w");
    for (int i = 0; i < 4; i++) {
        fprintf(fp, "%d, %f, %f\n", lst[i], rate1[i],rate2[i]);
    }
}

double multiply(int N){
    int *A = malloc(N*N*sizeof(int));
    int *B = malloc(N*N*sizeof(int));
    int *C = malloc(N*N*sizeof(int));

    #define IDX(i,j) ((i)*N + (j))

    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) {
            A[IDX(i,j)] = (i==j)?1:2;
            B[IDX(i,j)] = (i<j)?10:20;
            C[IDX(i,j)] = 0;
        }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[IDX(i,j)] += A[IDX(i,k)] * B[IDX(k,j)];
    gettimeofday(&end, NULL);
    free(A); free(B); free(C);
    // printf("time=%f\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6);
    double running_time = (end.tv_sec - start.tv_sec)+(end.tv_usec - start.tv_usec) * 1e-6;
    return running_time;
    

};

double multiply_tile(int N){

    int *A = malloc(N*N*sizeof(int));
    int *B = malloc(N*N*sizeof(int));
    int *C = malloc(N*N*sizeof(int));

    #define IDX(i,j) ((i)*N + (j))

    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) {
            A[IDX(i,j)] = (i==j)?1:2;
            B[IDX(i,j)] = (i<j)?10:20;
            C[IDX(i,j)] = 0;
        }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int ii = 0; ii < N; ii += BS)
        for (int jj = 0; jj < N; jj += BS)
            for (int kk = 0; kk < N; kk += BS)
                for (int i = ii; i < ii+BS && i < N; i++)
                    for (int j = jj; j < jj+BS && j < N; j++)
                        for (int k = kk; k < kk+BS && k < N; k++)
                            C[IDX(i,j)] += A[IDX(i,k)] * B[IDX(k,j)];
    
    gettimeofday(&end, NULL);
    free(A); free(B); free(C);
    // printf("time=%f\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6);
    double running_time = (end.tv_sec - start.tv_sec)+(end.tv_usec - start.tv_usec) * 1e-6;
    return running_time;

}

