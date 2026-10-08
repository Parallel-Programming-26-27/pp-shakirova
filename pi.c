#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double pi_sequential(long long n) {
    double sum = 0.0;
    for (long long i = 0; i < n; i++) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * i + 1.0);
    }
    return 4.0 * sum;
}

double pi_parallel(long long n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * i + 1.0);
    }
    return 4.0 * sum;
}

int count_threads() {
    int used = 1;
    #pragma omp parallel
    {
        #pragma omp single
        used = omp_get_num_threads();
    }
    return used;
}

int main(int argc, char *argv[]) {
    long long n = 1000000000LL; 
    int threads = omp_get_max_threads();

    if (argc > 1) n = atoll(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    if (n <= 0 || threads <= 0) {
        fprintf(stderr, "Использование: %s [n] [threads]\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(threads);

    double t_seq = omp_get_wtime();
    double pi_seq = pi_sequential(n);
    t_seq = omp_get_wtime() - t_seq;

    double t_par = omp_get_wtime();
    double pi_par = pi_parallel(n);
    t_par = omp_get_wtime() - t_par;

    int used_threads = count_threads();

    printf("n = %lld\n\n", n);

    printf("Последовательная версия:\n");
    printf("  pi      = %.12f\n", pi_seq);
    printf("  ошибка  = %.3e\n", fabs(pi_seq - M_PI));
    printf("  время   = %.4f с\n", t_seq);
    printf("  потоков = 1\n\n");

    printf("OpenMP версия:\n");
    printf("  pi      = %.12f\n", pi_par);
    printf("  ошибка  = %.3e\n", fabs(pi_par - M_PI));
    printf("  время   = %.4f с\n", t_par);
    printf("  потоков = %d\n\n", used_threads);

    printf("Ускорение = %.2f\n", t_seq / t_par);

    return 0;
}