#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define REPS 5

void ijk_seq(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void ijk_omp(const double *A, const double *B, double *C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void ikj_seq(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void ikj_omp(const double *A, const double *B, double *C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void blocked_omp(const double *A, const double *B, double *C, int N, int bs) {
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += bs) {
        for (int jj = 0; jj < N; jj += bs) {
            for (int kk = 0; kk < N; kk += bs) {
                for (int i = ii; i < ii + bs && i < N; i++) {
                    for (int k = kk; k < kk + bs && k < N; k++) {
                        double a = A[i * N + k];
                        for (int j = jj; j < jj + bs && j < N; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

double run(int variant, const double *A, const double *B, double *C, int N, int bs) {
    memset(C, 0, (size_t)N * N * sizeof *C);
    double t = omp_get_wtime();
    switch (variant) {
        case 0: ijk_seq(A, B, C, N); break;
        case 1: ijk_omp(A, B, C, N); break;
        case 2: ikj_seq(A, B, C, N); break;
        case 3: ikj_omp(A, B, C, N); break;
        default: blocked_omp(A, B, C, N, bs);
    }
    return omp_get_wtime() - t;
}

double median_time(int variant, const double *A, const double *B, double *C, int N, int bs) {
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        t[r] = run(variant, A, B, C, N, bs);
    }
    for (int p = 1; p < REPS; p++) {
        double x = t[p];
        int q = p - 1;
        while (q >= 0 && t[q] > x) {
            t[q + 1] = t[q];
            q--;
        }
        t[q + 1] = x;
    }
    return t[REPS / 2];
}

int main(int argc, char *argv[]) {
    int N = 512;
    int bs = 32;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) bs = atoi(argv[2]);

    if (N <= 0 || bs <= 0) {
        fprintf(stderr, "Использование: OMP_NUM_THREADS=p %s [N] [размер_блока]\n", argv[0]);
        return 1;
    }

    size_t size = (size_t)N * N * sizeof(double);
    double *A = malloc(size);
    double *B = malloc(size);
    double *C = malloc(size);
    double *R = malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i + j) % 10 / 10.0;
            B[i * N + j] = (2 * i + j) % 10 / 10.0;
        }
    }

    const char *names[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};
    double t[5];
    int threads = omp_get_max_threads();

    printf("N = %d, размер блока = %d, потоков = %d, замеров = %d (медиана)\n\n", N, bs, threads, REPS);

    for (int v = 0; v < 5; v++) {
        t[v] = median_time(v, A, B, C, N, bs);
        if (v == 0) {
            memcpy(R, C, size);
        }
        double diff = 0.0;
        for (size_t x = 0; x < (size_t)N * N; x++) {
            double d = fabs(C[x] - R[x]);
            if (d > diff) {
                diff = d;
            }
        }
        printf("%-12s время = %.6f с, отклонение от ijk_seq = %.2e, проверка: %s\n",
               names[v], t[v], diff, diff < 1e-9 ? "успешно" : "ОШИБКА");
    }

    printf("\nУскорение ijk = %.2f, эффективность = %.2f\n", t[0] / t[1], t[0] / t[1] / threads);
    printf("Ускорение ikj = %.2f, эффективность = %.2f\n", t[2] / t[3], t[2] / t[3] / threads);
    printf("Отношение ikj_seq / blocked_omp = %.2f\n", t[2] / t[4]);

    free(A);
    free(B);
    free(C);
    free(R);
    return 0;
}