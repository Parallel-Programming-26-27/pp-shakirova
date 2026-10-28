#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define REPS 5
#define EPS 1e-6
#define TOL 1e-8

double soft(double z, double a) {
    if (z > a) return z - a;
    if (z < -a) return z + a;
    return 0.0;
}

void update(double *w, double *b, const double *g, double gb, int N, int d, double lambda, double eta) {
    for (int j = 0; j < d; j++) {
        w[j] = soft(w[j] - eta * g[j] / N, eta * lambda);
    }
    *b -= eta * gb / N;
}

void train_serial(const double *X, const double *y, double *w, double *b, double *g,
                  int N, int d, double lambda, double eta, int K) {
    for (int k = 0; k < K; k++) {
        memset(g, 0, d * sizeof(double));
        double gb = 0.0;
        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * d;
            double pred = *b;
            for (int j = 0; j < d; j++) {
                pred += x[j] * w[j];
            }
            double e = pred - y[i];
            for (int j = 0; j < d; j++) {
                g[j] += e * x[j];
            }
            gb += e;
        }
        update(w, b, g, gb, N, d, lambda, eta);
    }
}

/* OpenMP: объекты делятся между потоками. reduction дает каждому потоку свои копии g и gb,
   в конце они складываются, поэтому гонок данных нет */
void train_omp(const double *X, const double *y, double *w, double *b, double *g,
               int N, int d, double lambda, double eta, int K) {
    for (int k = 0; k < K; k++) {
        memset(g, 0, d * sizeof(double));
        double gb = 0.0;
        #pragma omp parallel for reduction(+:g[:d], gb)
        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * d;
            double pred = *b;
            for (int j = 0; j < d; j++) {
                pred += x[j] * w[j];
            }
            double e = pred - y[i];
            for (int j = 0; j < d; j++) {
                g[j] += e * x[j];
            }
            gb += e;
        }
        update(w, b, g, gb, N, d, lambda, eta);
    }
}

/* SIMD: потоков нет, циклы по признакам j векторизуются (omp simd);
   скалярное произведение - simd-редукция */
void train_simd(const double *X, const double *y, double *w, double *b, double *g,
                int N, int d, double lambda, double eta, int K) {
    for (int k = 0; k < K; k++) {
        memset(g, 0, d * sizeof(double));
        double gb = 0.0;
        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * d;
            double pred = *b;
            #pragma omp simd reduction(+:pred)
            for (int j = 0; j < d; j++) {
                pred += x[j] * w[j];
            }
            double e = pred - y[i];
            #pragma omp simd
            for (int j = 0; j < d; j++) {
                g[j] += e * x[j];
            }
            gb += e;
        }
        update(w, b, g, gb, N, d, lambda, eta);
    }
}

/* OpenMP + SIMD: потоки делят объекты, внутри объекта циклы по j векторизуются */
void train_omp_simd(const double *X, const double *y, double *w, double *b, double *g,
                    int N, int d, double lambda, double eta, int K) {
    for (int k = 0; k < K; k++) {
        memset(g, 0, d * sizeof(double));
        double gb = 0.0;
        #pragma omp parallel for reduction(+:g[:d], gb)
        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * d;
            double pred = *b;
            #pragma omp simd reduction(+:pred)
            for (int j = 0; j < d; j++) {
                pred += x[j] * w[j];
            }
            double e = pred - y[i];
            #pragma omp simd
            for (int j = 0; j < d; j++) {
                g[j] += e * x[j];
            }
            gb += e;
        }
        update(w, b, g, gb, N, d, lambda, eta);
    }
}

double loss(const double *X, const double *y, const double *w, double b, int N, int d, double lambda) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        double pred = b;
        for (int j = 0; j < d; j++) {
            pred += X[(size_t)i * d + j] * w[j];
        }
        sum += (pred - y[i]) * (pred - y[i]);
    }
    double l1 = 0.0;
    for (int j = 0; j < d; j++) {
        l1 += fabs(w[j]);
    }
    return sum / (2.0 * N) + lambda * l1;
}

double run(int variant, const double *X, const double *y, double *w, double *b, double *g,
           int N, int d, double lambda, double eta, int K) {
    memset(w, 0, d * sizeof(double));
    *b = 0.0;
    double t = omp_get_wtime();
    switch (variant) {
        case 0: train_serial(X, y, w, b, g, N, d, lambda, eta, K); break;
        case 1: train_omp(X, y, w, b, g, N, d, lambda, eta, K); break;
        case 2: train_simd(X, y, w, b, g, N, d, lambda, eta, K); break;
        default: train_omp_simd(X, y, w, b, g, N, d, lambda, eta, K);
    }
    return omp_get_wtime() - t;
}

double median_time(int variant, const double *X, const double *y, double *w, double *b, double *g,
                   int N, int d, double lambda, double eta, int K) {
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        t[r] = run(variant, X, y, w, b, g, N, d, lambda, eta, K);
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
    int N = 100000;
    int d = 32;
    double lambda = 1e-3;
    double eta = 0.5;
    int K = 100;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) d = atoi(argv[2]);
    if (argc > 3) lambda = atof(argv[3]);
    if (argc > 4) eta = atof(argv[4]);
    if (argc > 5) K = atoi(argv[5]);

    if (N <= 0 || d <= 0 || lambda < 0 || eta <= 0 || K <= 0) {
        fprintf(stderr, "Использование: OMP_NUM_THREADS=p %s [N] [d] [lambda] [eta] [K]\n", argv[0]);
        return 1;
    }

    double *X = malloc((size_t)N * d * sizeof(double));
    double *y = malloc(N * sizeof(double));
    double *w = malloc(d * sizeof(double));
    double *g = malloc(d * sizeof(double));
    double *w_serial = malloc(d * sizeof(double));
    double b;

    const double w_true[4] = {1.0, 0.05, 0.005, 0.0};
    srand(42);
    for (size_t k = 0; k < (size_t)N * d; k++) {
        X[k] = 2.0 * rand() / RAND_MAX - 1.0;
    }
    for (int i = 0; i < N; i++) {
        double s = 0.5 + 0.1 * (2.0 * rand() / RAND_MAX - 1.0);
        for (int j = 0; j < d; j++) {
            s += w_true[j % 4] * X[(size_t)i * d + j];
        }
        y[i] = s;
    }

    const char *names[] = {"serial", "omp", "simd", "omp_simd"};
    double t_serial = 0.0;
    double loss_serial = 0.0;

    printf("N = %d, d = %d, lambda = %g, eta = %g, K = %d, потоков = %d, замеров = %d (медиана), допуск = %g\n\n",
           N, d, lambda, eta, K, omp_get_max_threads(), REPS, TOL);

    for (int v = 0; v < 4; v++) {
        double t = median_time(v, X, y, w, &b, g, N, d, lambda, eta, K);
        double f = loss(X, y, w, b, N, d, lambda);

        if (v == 0) {
            t_serial = t;
            loss_serial = f;
            memcpy(w_serial, w, d * sizeof(double));
        }

        double dw = 0.0;
        int zeros = 0;
        for (int j = 0; j < d; j++) {
            if (fabs(w[j] - w_serial[j]) > dw) {
                dw = fabs(w[j] - w_serial[j]);
            }
            if (fabs(w[j]) < EPS) {
                zeros++;
            }
        }
        int ok = dw < TOL && fabs(f - loss_serial) < TOL;

        printf("%-9s время = %.6f с, ускорение = %.2f, F = %.9f, Δw = %.2e, |w_j| < %g: %d, проверка: %s\n",
               names[v], t, t_serial / t, f, dw, EPS, zeros, ok ? "успешно" : "ОШИБКА");
    }

    free(X);
    free(y);
    free(w);
    free(g);
    free(w_serial);
    return 0;
}