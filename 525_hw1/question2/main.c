#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main() 
{
    struct timeval start, end;
    size_t length = 10000000;
    float *record = (float *)malloc(100 * sizeof(float));
    
    for (size_t skip = 1; skip < 101; skip++) //for part 1, i=1, and part 2, i>=2
    { 
        long sum = 0;
        long *a = (long *)malloc(skip * length * sizeof(long));
        for (size_t i = 0; i < skip * length; i++) {
            a[i] = 2;
        }
        

        gettimeofday(&start, NULL);
        for (size_t i = 0; i < skip * length; i += skip) {
            sum += a[i];
        }
        gettimeofday(&end, NULL);
    
        double t = (end.tv_sec - start.tv_sec)
                 + (end.tv_usec - start.tv_usec) * 1e-6;
        printf("skip=%zu, time=%f\n", skip, t);
        record[skip] = t;
        free(a);
    }
    FILE *fp = fopen("record.csv", "w");
    for (int i = 1; i < 101; i++) {
        fprintf(fp, "%d, %f\n", i, record[i]); // for part 3 
    }
    fclose(fp);
    return 0;
}