#include <stdio.h>
#include <sys/time.h>

int main() {
    static float a[10000000];
    for (int i = 0; i < 10000000; i++) a[i] = 1.0;

    float sum = 0;
    struct timeval start, end;

    gettimeofday(&start, NULL);
    for (int i = 0; i < 10000000; i++)
        sum += a[i];
    gettimeofday(&end, NULL);

    printf("Segment 1 time: %f\n",(end.tv_sec - start.tv_sec) +(end.tv_usec - start.tv_usec) * 1e-6);

    float s1=0, s2=0, s3=0, s4=0;
    gettimeofday(&start, NULL);
    for (int i = 0; i < 10000000; i += 4) {
        s1 += a[i];
        s2 += a[i+1];
        s3 += a[i+2];
        s4 += a[i+3];
    }
    sum = s1+s2+s3+s4;
    gettimeofday(&end, NULL);

    printf("Segment 2 time: %f\n", (end.tv_sec - start.tv_sec) +(end.tv_usec - start.tv_usec) * 1e-6);
}
