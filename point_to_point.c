#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    int rank, size;
    const int TAG_MESSAGE = 0;
    const int TAG_REPLY = 1;
    char message[100];
    char reply[] = "Message received";
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("At least 2 processes are required.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            snprintf(message, sizeof(message), "Hello from rank 0 to rank %d", r);
            MPI_Send(message, (int)strlen(message) + 1, MPI_CHAR, r, TAG_MESSAGE, MPI_COMM_WORLD);
            printf("Rank 0 sent to rank %d: %s\n", r, message);
        }

        for (int r = 1; r < size; r++) {
            MPI_Recv(message, sizeof(message), MPI_CHAR, r, TAG_REPLY, MPI_COMM_WORLD, &status);
            printf("Rank 0 received from rank %d: %s\n", r, message);
        }
    } else {
        MPI_Recv(message, sizeof(message), MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        printf("Rank %d received from rank 0: %s\n", rank, message);

        MPI_Send(reply, sizeof(reply), MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
        printf("Rank %d sent to rank 0: %s\n", rank, reply);
    }

    if (rank == 0 || rank == 1) {
        int lengths[] = {1, 1024, 65536, 1048576};
        static char buffer[1048576];
        int warmup = 10;
        int repeats = 100;

        for (int i = 0; i < 1048576; i++) {
            buffer[i] = 'A';
        }

        if (rank == 0) {
            printf("Size (bytes) | Repetitions | Average time (s)\n");
        }

        for (int i = 0; i < 4; i++) {
            int count = lengths[i];
            double start = 0;

            for (int j = 0; j < warmup + repeats; j++) {
                if (rank == 0) {
                    if (j == warmup) {
                        start = MPI_Wtime();
                    }
                    MPI_Send(buffer, count, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
                    MPI_Recv(buffer, count, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, &status);
                } else {
                    MPI_Recv(buffer, count, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
                    MPI_Send(buffer, count, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
                }
            }

            if (rank == 0) {
                printf("%d | %d | %.9f\n", count, repeats, (MPI_Wtime() - start) / repeats);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
