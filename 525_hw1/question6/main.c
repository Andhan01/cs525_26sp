#include <stdio.h>
#include <sys/time.h>

int main() {
    static float a[1000000];
    for (int i = 0; i < 1000000; i++) a[i] = i;

    struct timeval start, end;

    gettimeofday(&start, NULL);
    for (int i = 0; i < 1000000; i++)
        a[i] = a[i] / 2.0;
    gettimeofday(&end, NULL);

    printf("div time: %f\n",
        (end.tv_sec - start.tv_sec) +
        (end.tv_usec - start.tv_usec) * 1e-6);

    float one_over_two = 1.0/2.0;
    gettimeofday(&start, NULL);
    for (int i = 0; i < 1000000; i++)
        a[i] = a[i] * one_over_two;
    gettimeofday(&end, NULL);

    printf("mul time: %f\n",
        (end.tv_sec - start.tv_sec) +
        (end.tv_usec - start.tv_usec) * 1e-6);
}
