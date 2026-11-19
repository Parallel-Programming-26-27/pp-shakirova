#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n, m;
    int *offsets, *edges;
} Graph;

static void sequential_bfs(Graph *g, int source, int *dist) {
    int *queue = malloc((size_t)g->n * sizeof(int));
    int head = 0, tail = 0;
    for (int i = 0; i < g->n; i++) dist[i] = -1;
    dist[source] = 0;
    queue[tail++] = source;
    while (head < tail) {
        int v = queue[head++];
        for (int j = g->offsets[v]; j < g->offsets[v + 1]; j++) {
            int u = g->edges[j];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
    free(queue);
}

static void parallel_bfs(Graph *g, int source, int rank, int size, int sparse,
                         int *dist, double *time, double *exchange,
                         long long *processed, int *levels, int *max_frontier) {
    int n = g->n;
    int *frontier = malloc((size_t)n * sizeof(int));
    int *next = malloc((size_t)n * sizeof(int));
    int *local = calloc((size_t)n, sizeof(int));
    int *global = calloc((size_t)n, sizeof(int));
    int threads = omp_get_max_threads();
    int **lists = calloc((size_t)threads, sizeof(int *));
    int *lengths = calloc((size_t)threads, sizeof(int));
    int *counts = malloc((size_t)size * sizeof(int));
    int *displs = malloc((size_t)size * sizeof(int));
    if (!frontier || !next || !local || !global || !lists || !lengths || !counts || !displs)
        MPI_Abort(MPI_COMM_WORLD, 1);

    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[source] = 0;
    frontier[0] = source;
    int count = 1;
    *levels = 0;
    *max_frontier = 0;
    *processed = 0;
    *exchange = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    while (count > 0) {
        if (count > *max_frontier) *max_frontier = count;
        if (rank == 0 && !sparse)
            printf("Frontier level=%d size=%d\n", *levels, count);
        memset(local, 0, (size_t)n * sizeof(int));
        memset(lengths, 0, (size_t)threads * sizeof(int));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int cap = 64, len = 0;
            int *found = malloc((size_t)cap * sizeof(int));
            if (!found) MPI_Abort(MPI_COMM_WORLD, 1);

            #pragma omp for schedule(runtime)
            for (int k = rank; k < count; k += size) {
                int v = frontier[k];
                for (int j = g->offsets[v]; j < g->offsets[v + 1]; j++) {
                    int u = g->edges[j];
                    if (dist[u] == -1) {
                        if (len == cap) {
                            cap *= 2;
                            int *tmp = realloc(found, (size_t)cap * sizeof(int));
                            if (!tmp) MPI_Abort(MPI_COMM_WORLD, 1);
                            found = tmp;
                        }
                        found[len++] = u;
                    }
                }
            }
            lists[tid] = found;
            lengths[tid] = len;
        }

        int local_count = 0;
        int *local_list = malloc((size_t)n * sizeof(int));
        if (!local_list) MPI_Abort(MPI_COMM_WORLD, 1);
        for (int t = 0; t < threads; t++) {
            for (int j = 0; j < lengths[t]; j++) {
                int u = lists[t][j];
                if (!local[u]) {
                    local[u] = 1;
                    local_list[local_count++] = u;
                }
            }
            free(lists[t]);
        }
        *processed += (count + size - rank - 1) / size;

        double comm_start = MPI_Wtime();
        if (!sparse) {
            MPI_Allreduce(local, global, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        } else {
            MPI_Allgather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
            int total = 0;
            for (int r = 0; r < size; r++) {
                displs[r] = total;
                total += counts[r];
            }
            int *received = malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
            if (!received) MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Allgatherv(local_list, local_count, MPI_INT,
                           received, counts, displs, MPI_INT, MPI_COMM_WORLD);
            memset(global, 0, (size_t)n * sizeof(int));
            for (int j = 0; j < total; j++) global[received[j]] = 1;
            free(received);
        }
        *exchange += MPI_Wtime() - comm_start;
        free(local_list);

        int next_count = 0;
        for (int u = 0; u < n; u++) {
            if (global[u] && dist[u] == -1) {
                dist[u] = *levels + 1;
                next[next_count++] = u;
            }
        }
        int *tmp = frontier;
        frontier = next;
        next = tmp;
        count = next_count;
        (*levels)++;
    }
    double local_time = MPI_Wtime() - start;
    MPI_Reduce(&local_time, time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double max_exchange = 0;
    MPI_Reduce(exchange, &max_exchange, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    *exchange = max_exchange;

    free(frontier);
    free(next);
    free(local);
    free(global);
    free(lists);
    free(lengths);
    free(counts);
    free(displs);
}

int main(int argc, char **argv) {
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);

    if (argc < 3) {
        if (rank == 0) printf("Usage: ./bfs graph.txt source [directed]\n"
                              "   or: ./bfs gen vertices degree source\n");
        MPI_Finalize();
        return 1;
    }

    Graph g = {0};
    int *from = NULL, *to = NULL;
    int source, directed = 0;
    if (strcmp(argv[1], "gen") == 0) {
        if (argc < 5) MPI_Abort(MPI_COMM_WORLD, 1);
        g.n = atoi(argv[2]);
        int degree = atoi(argv[3]);
        source = atoi(argv[4]);
        if (g.n < 2 || degree < 2 || degree % 2) MPI_Abort(MPI_COMM_WORLD, 1);
        g.m = g.n * (degree / 2);
        from = malloc((size_t)g.m * sizeof(int));
        to = malloc((size_t)g.m * sizeof(int));
        int at = 0;
        for (int v = 0; v < g.n; v++) {
            for (int j = 0; j < degree / 2; j++) {
                int step = j == 0 ? 1 : 1 + (int)(((unsigned)v * 1103515245u +
                                  (unsigned)j * 12345u) % (unsigned)(g.n - 1));
                from[at] = v;
                to[at++] = (v + step) % g.n;
            }
        }
    } else {
        FILE *file = fopen(argv[1], "r");
        source = atoi(argv[2]);
        if (argc > 3 && strcmp(argv[3], "directed") == 0) directed = 1;
        if (!file || fscanf(file, "%d %d", &g.n, &g.m) != 2 || g.n <= 0 || g.m < 0)
            MPI_Abort(MPI_COMM_WORLD, 1);
        from = malloc((size_t)(g.m > 0 ? g.m : 1) * sizeof(int));
        to = malloc((size_t)(g.m > 0 ? g.m : 1) * sizeof(int));
        for (int i = 0; i < g.m; i++) {
            if (fscanf(file, "%d %d", &from[i], &to[i]) != 2 ||
                from[i] < 0 || from[i] >= g.n || to[i] < 0 || to[i] >= g.n)
                MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(file);
    }
    if (!from || !to || source < 0 || source >= g.n) MPI_Abort(MPI_COMM_WORLD, 1);

    g.offsets = calloc((size_t)g.n + 1, sizeof(int));
    int *pos = malloc((size_t)g.n * sizeof(int));
    int arcs = g.m * (directed ? 1 : 2);
    g.edges = malloc((size_t)(arcs > 0 ? arcs : 1) * sizeof(int));
    if (!g.offsets || !g.edges || !pos) MPI_Abort(MPI_COMM_WORLD, 1);
    for (int i = 0; i < g.m; i++) {
        g.offsets[from[i] + 1]++;
        if (!directed) g.offsets[to[i] + 1]++;
    }
    for (int v = 0; v < g.n; v++) g.offsets[v + 1] += g.offsets[v];
    for (int v = 0; v < g.n; v++) pos[v] = g.offsets[v];
    for (int i = 0; i < g.m; i++) {
        g.edges[pos[from[i]]++] = to[i];
        if (!directed) g.edges[pos[to[i]]++] = from[i];
    }
    free(from);
    free(to);
    free(pos);

    int *seq = malloc((size_t)g.n * sizeof(int));
    int *dist = malloc((size_t)g.n * sizeof(int));
    if (!seq || !dist) MPI_Abort(MPI_COMM_WORLD, 1);
    double seq_time = 0;
    if (rank == 0) {
        double start = MPI_Wtime();
        sequential_bfs(&g, source, seq);
        seq_time = MPI_Wtime() - start;
        printf("Graph V=%d E=%d AverageDegree=%.2f Source=%d MPI=%d OMP=%d Sequential=%.9f s\n",
               g.n, g.m, (double)arcs / g.n, source, size, omp_get_max_threads(), seq_time);
    }

    for (int mode = 0; mode < 2; mode++) {
        double time = 0, exchange = 0;
        long long processed = 0;
        int levels = 0, max_frontier = 0;
        parallel_bfs(&g, source, rank, size, mode, dist, &time, &exchange,
                     &processed, &levels, &max_frontier);
        long long *work = rank == 0 ? malloc((size_t)size * sizeof(long long)) : NULL;
        MPI_Gather(&processed, 1, MPI_LONG_LONG_INT,
                   work, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            int mismatch = 0, unreachable = 0;
            for (int v = 0; v < g.n; v++) {
                if (seq[v] != dist[v]) mismatch++;
                if (dist[v] == -1) unreachable++;
            }
            double speedup = seq_time / time;
            printf("Mode=%s Levels=%d MaxFrontier=%d Unreachable=%d Mismatch=%d Time=%.9f s Exchange=%.9f s Speedup=%.4f Efficiency=%.4f Correct=%s\n",
                   mode ? "Allgatherv" : "Allreduce", levels, max_frontier,
                   unreachable, mismatch, time, exchange, speedup,
                   speedup / (size * omp_get_max_threads()), mismatch ? "No" : "Yes");
            printf("Processed per MPI rank:");
            for (int r = 0; r < size; r++) printf(" %lld", work[r]);
            printf("\n");
        }
        free(work);
    }

    free(seq);
    free(dist);
    free(g.offsets);
    free(g.edges);
    MPI_Finalize();
    return 0;
}
