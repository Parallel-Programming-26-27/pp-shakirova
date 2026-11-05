#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size, n = 0;
    int *data = NULL, *local_data;
    int *counts = NULL, *displs = NULL;
    long long local_sum = 0, gather_sum = 0, reduce_sum = 0, allreduce_sum = 0;
    long long *sums = NULL;
    double start = 0, elapsed = 0;
    double gather_time = 0, reduce_time = 0;
    int repeats = 1000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0 && argc > 1) {
        n = atoi(argv[1]);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        if (rank == 0) {
            printf("N must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // дополнительные задания: MPI_Scatterv и MPI_Allreduce для всех процессов

    int local_n = n / size + (rank < n % size);
    local_data = malloc((local_n > 0 ? local_n : 1) * sizeof(int));

    if (rank == 0) {
        data = malloc(n * sizeof(int));
        sums = malloc(size * sizeof(long long));

        if (n % size != 0) {
            counts = malloc(size * sizeof(int));
            displs = malloc(size * sizeof(int));

            if (counts == NULL || displs == NULL) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i = 0; i < size; i++) {
                counts[i] = n / size + (i < n % size);
                displs[i] = i * (n / size) + (i < n % size ? i : n % size);
            }
        }
    }

    if (local_data == NULL ||
        (rank == 0 && (data == NULL || sums == NULL))) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            data[i] = i + 1;
        }
        start = MPI_Wtime();
    }

    if (n % size == 0) {
        MPI_Scatter(data, local_n, MPI_INT,
                    local_data, local_n, MPI_INT,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(data, counts, displs, MPI_INT,
                     local_data, local_n, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
    }

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT,
               sums, 1, MPI_LONG_LONG_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            gather_sum += sums[i];
        }
    }

    MPI_Reduce(&local_sum, &reduce_sum, 1,
               MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Allreduce(&local_sum, &allreduce_sum, 1,
                  MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        elapsed = MPI_Wtime() - start;
        long long expected = (long long)n * (n + 1) / 2;

        printf("P=%d N=%d Expected=%lld Gather=%lld Reduce=%lld Allreduce=%lld Correct=%s Time=%.9f s\n",
               size, n, expected, gather_sum, reduce_sum, allreduce_sum,
               gather_sum == expected && reduce_sum == expected &&
               allreduce_sum == expected ? "Yes" : "No", elapsed);
    }

    printf("Rank %d Allreduce=%lld\n", rank, allreduce_sum);

    // эксперимент 3.4

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start = MPI_Wtime();
    }

    for (int j = 0; j < repeats; j++) {
        MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT,
                   sums, 1, MPI_LONG_LONG_INT,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            gather_sum = 0;
            for (int i = 0; i < size; i++) {
                gather_sum += sums[i];
            }
        }
    }

    if (rank == 0) {
        gather_time = (MPI_Wtime() - start) / repeats;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start = MPI_Wtime();
    }

    for (int j = 0; j < repeats; j++) {
        MPI_Reduce(&local_sum, &reduce_sum, 1,
                   MPI_LONG_LONG_INT, MPI_SUM,
                   0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        reduce_time = (MPI_Wtime() - start) / repeats;
        long long expected = (long long)n * (n + 1) / 2;

        printf("Aggregation P=%d N=%d Repetitions=%d Gather=%.9f s Reduce=%.9f s Correct=%s\n",
               size, n, repeats, gather_time, reduce_time,
               gather_sum == expected && reduce_sum == expected ? "Yes" : "No");
    }

    free(local_data);
    free(data);
    free(sums);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
