#include <stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda.h>
#include<math.h>
#include<iostream>

#define STB_IMAGE_IMPLEMENTATION
#include<stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//CPU Functions
void read_image(unsigned char* img_data, int &w, int &h, int &channels);
void cpu_blur_image(unsigned char* img, unsigned char* blurred_img, const int w, const int h);

// GPU Kernels
__global__ void gpu_blur_image(unsigned char* img, unsigned char* blurred_img, int w, int h);
__global__ void separateChannels(unsigned char* img, unsigned char* rimg, unsigned char* gimg, unsigned char* bimg, int w, int h);
__global__ void recombineChannels(unsigned char* blurRimg, unsigned char* blurGimg, unsigned char* blurBimg, unsigned char* blurredImage, int w, int h);
__global__ void recombineChannels(unsigned char* redChannel, unsigned char* greenChannel, unsigned char* blueChannel, unsigned char* outputImag, int w, int h);


int main(int argc, char* argv[])
{
    clock_t start, end;
    unsigned char* img_data;
    int width, height, channels;
    
    //****************************************************************
    //***************** Reading Image ********************************
    //****************************************************************

    img_data = stbi_load("pic.png", &width, &height, &channels, 4);
    if(img_data == NULL){
        printf("Error in loading image\n");
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels \n", width, height, channels);

    //****************************************************************
    //*************** Allocating Memory ******************************
    //****************************************************************
    unsigned char* d_imgData;

    // Host Variables for channel images
    unsigned char* h_redImage;
    unsigned char* h_greenImage;
    unsigned char* h_blueImage;

    unsigned char* h_blurRedImage;
    unsigned char* h_blurGreenImage;
    unsigned char* h_blurBlueImage;

    // Device Variables for channel images
    unsigned char* d_redImage;
    unsigned char* d_greenImage;
    unsigned char* d_blueImage;

    unsigned char* d_blurRedImage;
    unsigned char* d_blurGreenImage;
    unsigned char* d_blurBlueImage;

    h_redImage = (unsigned char*) malloc(width * height * sizeof(unsigned char));
    h_greenImage = (unsigned char*) malloc(width * height * sizeof(unsigned char));
    h_blueImage = (unsigned char*) malloc(width * height * sizeof(unsigned char));

    cudaMalloc(&d_imgData, sizeof(unsigned char) * width * height * channels);
    cudaMalloc(&d_redImage, sizeof(unsigned char) * width * height);
    cudaMalloc(&d_greenImage, sizeof(unsigned char) * width * height);
    cudaMalloc(&d_blueImage, sizeof(unsigned char) * width * height);

    //****************************************************************
    //************* Copying from host to GPU *************************
    //****************************************************************
    cudaMemcpy(d_imgData, img_data, sizeof(unsigned char) * height * width * channels, cudaMemcpyHostToDevice);

    //****************************************************************
    //************* Setting Block & Grid Dim *************************
    //****************************************************************
    const dim3 blockSize(16,16,1);
    const dim3 gridSize((int) ceil(width/16), (int) ceil(height)/16);

    //****************************************************************
    //************* Separating Img into Channels *********************
    //****************************************************************
    separateChannels<<<gridSize, blockSize>>>(d_imgData, d_redImage, d_greenImage, d_blueImage, width, height);
    cudaDeviceSynchronize();

    //****************************************************************
    //********* Copying Channel Imgs from device to host *************
    //****************************************************************
    cudaMemcpy(h_redImage, d_redImage, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_greenImage, d_greenImage, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blueImage, d_blueImage, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);
     
    //****************************************************************
    //*********************** Saving Images **************************
    //****************************************************************
    stbi_write_png("red_channel_image_gpu.png", width, height, 1, h_redImage, width);
    stbi_write_png("green_channel_image_gpu.png", width, height, 1, h_greenImage, width);
    stbi_write_png("blue_channel_image_gpu.png", width, height, 1, h_blueImage, width);

    //****************************************************************
    //***************** Freeing Host Channel Imgs ********************
    //****************************************************************
    free(h_redImage);
    free(h_greenImage);
    free(h_blueImage);

    //****************************************************************
    //*********************** Blurring Image *************************
    //****************************************************************
    h_blurRedImage = (unsigned char*) malloc(width * height * sizeof(unsigned char));
    h_blurGreenImage = (unsigned char*) malloc(width * height * sizeof(unsigned char));
    h_blurBlueImage = (unsigned char*) malloc(width * height * sizeof(unsigned char));

    cudaMalloc(&d_blurRedImage, sizeof(unsigned char) * width * height);
    cudaMalloc(&d_blurGreenImage, sizeof(unsigned char) * width * height);
    cudaMalloc(&d_blurBlueImage, sizeof(unsigned char) * width * height);

    gpu_blur_image<<<gridSize, blockSize>>>(d_redImage, d_blurRedImage, width, height);
    gpu_blur_image<<<gridSize, blockSize>>>(d_greenImage, d_blurGreenImage, width, height);
    gpu_blur_image<<<gridSize, blockSize>>>(d_blueImage, d_blurBlueImage, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_blurRedImage, d_blurRedImage, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blurGreenImage, d_blurGreenImage, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blurBlueImage, d_blurBlueImage, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    stbi_write_png("blurred_red_channel_image_gpu.png", width, height, 1, h_blurRedImage, width);
    stbi_write_png("blurred_green_channel_image_gpu.png", width, height, 1, h_blurGreenImage, width);
    stbi_write_png("blurred_blue_channel_image_gpu.png", width, height, 1, h_blurBlueImage, width);

    stbi_image_free(img_data);
    free(h_blurRedImage);
    free(h_blurGreenImage);
    free(h_blurBlueImage);
    cudaFree(d_redImage);
    cudaFree(d_greenImage);
    cudaFree(d_blueImage);

    unsigned char* h_blurred_image;
    unsigned char* d_blurred_image;
    h_blurred_image = (unsigned char*) malloc(width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_blurred_image, sizeof(unsigned char) * width * height * channels);

    recombineChannels<<<gridSize, blockSize>>>(d_blurRedImage, d_blurGreenImage, d_blurBlueImage, d_blurred_image, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_blurred_image, d_blurred_image, sizeof(unsigned char) * width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_png("final_image_gpu.png", width, height, 4, h_blurred_image, width * channels);

    free(h_blurred_image);
    cudaFree(d_blurRedImage);
    cudaFree(d_blurGreenImage);
    cudaFree(d_blurBlueImage);
    cudaFree(d_blurred_image);

}


__global__ void separateChannels(unsigned char* d_img, unsigned char* d_rimg, unsigned char* d_gimg, unsigned char* d_bimg, int w, int h)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < w && row < h)
	{
        int single_offset = row * w + col;
        int rgb_offset = single_offset * 4;

        d_rimg[single_offset] += (unsigned char) d_img[rgb_offset];
        d_gimg[single_offset] += (unsigned char) d_img[rgb_offset + 1];
        d_bimg[single_offset] += (unsigned char) d_img[rgb_offset + 2];
    }
}

__global__ void gpu_blur_image(unsigned char* img, unsigned char* blurred_img, int w, int h)
{
    int BLUR_SIZE = 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < w && row < h)
    {
        int pixVal = 0;
        float pixels = 0;

        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; blurRow++)
        {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if(curRow > -1 && curRow < h && curCol > -1 && curCol < w)
                {
                    pixVal += (float) img[curRow * w + curCol];
                    pixels++;
                }
            }
        }
        blurred_img[row * w + col] = (unsigned char) (pixVal / pixels);
    }
}


__global__ void recombineChannels(unsigned char* redChannel, unsigned char* greenChannel, unsigned char* blueChannel, unsigned char* outputImag, int w, int h)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

   if (col < w && row < h)
	{
        int single_offset = row * w + col;
        int rgb_offset = single_offset * 4;

        outputImag[rgb_offset] += (unsigned char) redChannel[single_offset];
        outputImag[rgb_offset + 1] += (unsigned char) greenChannel[single_offset];
        outputImag[rgb_offset + 2] += (unsigned char) blueChannel[single_offset];
        outputImag[rgb_offset + 3] += (unsigned char) 255;
    }
}
