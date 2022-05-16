#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<iostream>
#include<time.h>

#define STB_IMAGE_IMPLEMENTATION
#include<stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void cpu_c_to_g(unsigned char* rgb, unsigned char* gray, int w, int h);
__global__ void gpu_c_to_g(unsigned char* rgb, unsigned char* gray, int w, int h);

int main(int argc, char* argv[])
{
    int width, height, channels;
    unsigned char* gray_img; 
    clock_t start, end;

    // Reading Img
    unsigned char* img_data = stbi_load("pic.png", &width, &height, &channels, 0);
    if(img_data == NULL){
        printf("Error in loading image\n");
        return EXIT_FAILURE;
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels \n", width, height, channels);

    
    // Allocating Memory for Img Array
    if( !(gray_img = (unsigned char*) malloc(width * height * sizeof(unsigned char))))
	{
	   printf("Error allocating array h_gray\n");
	   return EXIT_FAILURE;
	}

    // Compute CPU running time
    start = clock();
    cpu_c_to_g(img_data, gray_img, width, height);
    end = clock();
    printf("Total time taken by the sequential part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Computing GPU running time
    // Declaring device and host arrays
    unsigned char* d_color_img;
    unsigned char* d_gray_img;
    unsigned char* h_gray;

    h_gray = (unsigned char*) malloc(width * height * sizeof(unsigned char));

    // Allocating Memory for device
    cudaMalloc(&d_color_img, sizeof(unsigned char) * width * height * 4);
    cudaMalloc(&d_gray_img, sizeof(unsigned char) * width * height);
    cudaMemcpy(d_color_img, img_data, sizeof(unsigned char) * height * width * 4, cudaMemcpyHostToDevice);

    // Definition of grid and block dimensions
    const dim3 grid_size((int) ceil(width/16), (int) ceil(height)/16);
    const dim3 block_size(16, 16);

    // Execute Block and Grid Dimensions
    start = clock();
    gpu_c_to_g<<<grid_size, block_size>>>(d_color_img, d_gray_img, width, height);
    end = clock();
    printf("Total time taken by the parallel part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Copy result from device to host
    cudaMemcpy(h_gray, d_gray_img, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // Saving gray image
    stbi_write_png("grey_gpu.png", width, height, 1, h_gray, width);

    // Free Memory
    stbi_image_free(img_data);
    cudaFree(d_color_img);
    cudaFree(d_gray_img);
    return 0;
}

void cpu_c_to_g(unsigned char* rgb, unsigned char* gray, int w, int h)
{
    for(int j = 0; j < h; j++)
    {
        for(int i = 0; i < w; i++)
        {
            int gray_index = j * w + i;
            int rbg_index = 4 * gray_index;
            float r = (float) rgb[rbg_index];
            float g = (float) rgb[rbg_index + 1];
            float b = (float) rgb[rbg_index + 2];

            gray[gray_index] = (unsigned char) (r * 0.299f + g * 0.587f + b * 0.114f);
        }
    }
    stbi_write_png("grey_cpu.png", w, h, 1, gray, w);
}

__global__ void gpu_c_to_g(unsigned char* rgb, unsigned char* gray, int w, int h){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	//Compute for only those threads which map directly to 
	//image grid
	if (col < w && row < h)
	{
		int grey_offset = row * w + col;
		int rgb_offset = grey_offset * 4;
	
    	float r = (float) rgb[rgb_offset + 0];
	    float g = (float) rgb[rgb_offset + 1];
	    float b = (float) rgb[rgb_offset + 2];
	
	    gray[grey_offset] = (unsigned char) (r * 0.299f + g * 0.587f + b * 0.114f);
    }
}

