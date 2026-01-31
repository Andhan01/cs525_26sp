#include <stdio.h>
#include <sys/time.h>

int main() {
    int a[100];
    for (int i = 0; i < 100; i++) a[i] = i + 1;

    long sum = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);


    // //========part 1: not use the loop unrolling ====
    for (int i = 0; i < 100; i++) {
        sum += a[i];
    }
    // //==========end of part 1==================


    
    // //========part 2: use the loop unrolling ====
    // for (int i = 0; i < 100; i += 4) {
    //     sum += a[i] + a[i+1] + a[i+2] + a[i+3];
    // }
    // //==========end of part 2===================
    gettimeofday(&end, NULL);
    printf("sum=%ld, time=%f\n", sum,
        (end.tv_sec - start.tv_sec) +
        (end.tv_usec - start.tv_usec) * 1e-6);
}