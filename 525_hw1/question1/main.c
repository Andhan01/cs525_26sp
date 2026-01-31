#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main() {
    size_t length = 1000;//This is for part 1
    // size_t length = 10000000; // This is for part 2
    long sum = 0;
    long *a = (long *)malloc(length * sizeof(long));

    for (size_t i = 0; i < length; i++) {
        a[i] = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (size_t i = 0; i < length; i++) {
        sum += a[i];
    }

    gettimeofday(&end, NULL);

    double t = (end.tv_sec - start.tv_sec)
             + (end.tv_usec - start.tv_usec) * 1e-6;

    printf("length=%zu, time=%f seconds\n", length, t);
    printf("sum=%ld\n", sum);

    free(a);
    return 0;
}