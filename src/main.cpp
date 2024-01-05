#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <string>

using namespace cv;

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("usage: DisplayImage.out <Image_Path>\n");
		return -1;
	}
	Mat image;
	image = imread(argv[1], 1);
	if (!image.data)
	{
		printf("No image data \n");
		return -1;
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", image);
	waitKey(0);

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	// Print off a hello world message
	printf("Hello world from processor %s, rank %d out of %d processors\n",
		   processor_name, world_rank, world_size);

	// Finalize the MPI environment.
	MPI_Finalize();

#pragma omp parallel num_threads(2)
	{
		printf("Hello World from thread = %d\n", omp_get_thread_num());
	}
	return 0;
}
