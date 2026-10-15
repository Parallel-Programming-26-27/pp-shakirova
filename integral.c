#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double f(double x) {
    double c = cos(x);
    double s = sin(x);
    return c * c * c * s * s;
}

double integral_sequential(double a, double b, long long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (long long i = 0; i < n; i++) {
        sum += f(a + (i + 0.5) * h);
    }
    return h * sum;
}

double integral_parallel(double a, double b, long long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        sum += f(a + (i + 0.5) * h);
    }
    return h * sum;
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
    long long n = 100000000LL;
    int threads = omp_get_max_threads();
    double eps = 0.0;
    double a = 0.0;
    double b = M_PI / 3.0;

    if (argc > 1) n = atoll(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);
    if (argc > 3) eps = atof(argv[3]);
    if (argc > 4) a = atof(argv[4]);
    if (argc > 5) b = atof(argv[5]);

    if (n <= 0 || threads <= 0 || eps < 0 || b <= a) {
        fprintf(stderr, "Использование: %s [n] [threads] [eps] [a] [b]\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(threads);

    long long n_seq = n;
    double t_seq = omp_get_wtime();
    double res_seq = integral_sequential(a, b, n_seq);
    if (eps > 0) {
        double prev;
        do {
            prev = res_seq;
            n_seq = n_seq * 2;
            res_seq = integral_sequential(a, b, n_seq);
        } while (fabs(res_seq - prev) >= eps);
    }
    t_seq = omp_get_wtime() - t_seq;

    long long n_par = n;
    double t_par = omp_get_wtime();
    double res_par = integral_parallel(a, b, n_par);
    if (eps > 0) {
        double prev;
        do {
            prev = res_par;
            n_par = n_par * 2;
            res_par = integral_parallel(a, b, n_par);
        } while (fabs(res_par - prev) >= eps);
    }
    t_par = omp_get_wtime() - t_par;

    printf("Последовательная версия:\n");
    printf("  n        = %lld\n", n_seq);
    printf("  интеграл = %.12f\n", res_seq);
    printf("  время    = %.6f с\n", t_seq);
    printf("  потоков  = 1\n\n");

    printf("OpenMP версия:\n");
    printf("  n        = %lld\n", n_par);
    printf("  интеграл = %.12f\n", res_par);
    printf("  время    = %.6f с\n", t_par);
    printf("  потоков  = %d\n\n", count_threads());

    printf("Ускорение = %.2f\n", t_seq / t_par);

    return 0;
}