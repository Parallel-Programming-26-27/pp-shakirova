#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define REPS 5

typedef float elem;

void transpose_rows(const elem *A, elem *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

void transpose_cols(const elem *A, elem *B, int N) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

void transpose_blocked(const elem *A, elem *B, int N, int b) {
    for (int ii = 0; ii < N; ii += b) {
        for (int jj = 0; jj < N; jj += b) {
            for (int i = ii; i < ii + b && i < N; i++) {
                for (int j = jj; j < jj + b && j < N; j++) {
                    B[j * N + i] = A[i * N + j];
                }
            }
        }
    }
}

void transpose_rows_omp(const elem *A, elem *B, int N) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

void transpose_blocked_omp(const elem *A, elem *B, int N, int b) {
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii += b) {
        for (int jj = 0; jj < N; jj += b) {
            for (int i = ii; i < ii + b && i < N; i++) {
                for (int j = jj; j < jj + b && j < N; j++) {
                    B[j * N + i] = A[i * N + j];
                }
            }
        }
    }
}

void transpose_inplace(elem *A, int N, int b) {
    for (int ii = 0; ii < N; ii += b) {
        for (int jj = ii; jj < N; jj += b) {
            for (int i = ii; i < ii + b && i < N; i++) {
                for (int j = (jj == ii ? i + 1 : jj); j < jj + b && j < N; j++) {
                    elem tmp = A[i * N + j];
                    A[i * N + j] = A[j * N + i];
                    A[j * N + i] = tmp;
                }
            }
        }
    }
}

void transpose_inplace_omp(elem *A, int N, int b) {
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < N; ii += b) {
        for (int jj = ii; jj < N; jj += b) {
            for (int i = ii; i < ii + b && i < N; i++) {
                for (int j = (jj == ii ? i + 1 : jj); j < jj + b && j < N; j++) {
                    elem tmp = A[i * N + j];
                    A[i * N + j] = A[j * N + i];
                    A[j * N + i] = tmp;
                }
            }
        }
    }
}

void run(int version, const elem *A, elem *B, int N, int b) {
    switch (version) {
        case 0: transpose_rows(A, B, N); break;
        case 1: transpose_cols(A, B, N); break;
        case 2: transpose_blocked(A, B, N, b); break;
        case 3: transpose_rows_omp(A, B, N); break;
        case 4: transpose_blocked_omp(A, B, N, b); break;
        case 5: transpose_inplace(B, N, b); break;
        default: transpose_inplace_omp(B, N, b);
    }
}

int main(int argc, char *argv[]) {
    const char *names[] = {"rows", "cols", "blocked", "rows_omp", "blocked_omp", "inplace", "inplace_omp"};

    if (argc != 5) {
        fprintf(stderr, "Использование: %s N b потоки версия\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int b = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int version = -1;
    for (int v = 0; v < 7; v++) {
        if (strcmp(argv[4], names[v]) == 0) {
            version = v;
        }
    }

    if (N <= 0 || b <= 0 || threads <= 0 || version < 0) {
        fprintf(stderr, "Использование: %s N b потоки версия\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(threads);

    size_t count = (size_t)N * N;
    elem *A = malloc(count * sizeof(elem));
    elem *B = malloc(count * sizeof(elem));
    elem *R = malloc(count * sizeof(elem));

    for (size_t k = 0; k < count; k++) {
        A[k] = (elem)k;
    }
    transpose_rows(A, R, N);

    memcpy(B, A, count * sizeof(elem));
    run(version, A, B, N, b);

    double t[REPS];
    long mismatches = 0;
    for (int r = 0; r < REPS; r++) {
        memcpy(B, A, count * sizeof(elem));
        double start = omp_get_wtime();
        run(version, A, B, N, b);
        t[r] = omp_get_wtime() - start;
        for (size_t k = 0; k < count; k++) {
            if (B[k] != R[k]) {
                mismatches++;
            }
        }
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
    double median = t[REPS / 2];
    double bw = 2.0 * N * N * sizeof(elem) / median / 1e9;

    printf("N=%d b=%d threads=%d version=%s time=%.6f s BW=%.2f GB/s check=%s\n",
           N, b, threads, names[version], median, bw, mismatches == 0 ? "OK" : "FAIL");

    free(A);
    free(B);
    free(R);
    return 0;
}