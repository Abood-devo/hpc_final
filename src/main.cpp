#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::Mat image;
    int rows, cols;
    if (rank == 0) {
        image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE); 
        rows = image.rows;
        cols = image.cols;
    }
 
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = rows / size;

    int *send_counts = new int[size];
    int *displacements = new int[size];
    int displacement = 0;
    for (int i = 0; i < size; ++i) {
        send_counts[i] = rows_per_process * cols;
        displacements[i] = displacement;
        displacement += send_counts[i];
    }

    uchar *recv_buffer = new uchar[send_counts[rank]];

    MPI_Scatter(image.data, send_counts[rank], MPI_UNSIGNED_CHAR,
                recv_buffer, send_counts[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    cv::Mat process_image(rows_per_process, cols, CV_8UC1, recv_buffer);

    cv::Mat segmented_image;
    cv::threshold(process_image, segmented_image, 100, 255, cv::THRESH_BINARY);

    std::string process_filename = "process_" + std::to_string(rank) + "_segmented.jpg";
    cv::imwrite(process_filename, segmented_image);

    uchar *send_buffer = nullptr;
    if (rank == 0) {
        send_buffer = new uchar[rows * cols];
    }

    MPI_Gatherv(segmented_image.data, send_counts[rank], MPI_UNSIGNED_CHAR,
                send_buffer, send_counts, displacements, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cv::Mat final_segmented(rows, cols, CV_8UC1, send_buffer);
        std::string final_filename = "final_segmented_image.jpg";
        cv::imwrite(final_filename, final_segmented);
        // cv::imshow("Final Segmented Image", final_segmented);
        // cv::waitKey(0);
    }

    delete[] send_counts;
    delete[] displacements;
    delete[] recv_buffer;
    delete[] send_buffer;

    MPI_Finalize();
    return 0;
}
