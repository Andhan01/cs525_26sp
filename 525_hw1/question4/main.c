#include <stdio.h>
#include <sys/time.h>

// #define N 1000   // 改  100000 / 1000 / 10000

int main() {

    int lst[4] = {10,1000,10000,100000};
    for (int iter = 0; iter < 4; iter++){
        printf("start\n");
        int N = lst[iter];
        int a[N][N], b[N], result[N];
        for (int i = 0; i < N; i++) {
            b[i] = 3;
            result[i] = 0;
            for (int j = 0; j < N; j++)
                a[i][j] = 2;
        }

        struct timeval start, end;
        gettimeofday(&start, NULL);


        // //===========Original===================
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                result[i] += a[i][j] * b[j];
        // //===========End Original===============

        // //===========Interchanged===================
        // for (int j = 0; j < N; j++)
        //     for (int i = 0; i < N; i++)
        //         result[i] += a[i][j] * b[j];
        // //===========End Interchanged===============
        gettimeofday(&end, NULL);
        printf("time=%f\n",
            (end.tv_sec - start.tv_sec) +
            (end.tv_usec - start.tv_usec) * 1e-6);

    }
    return 0;
}
