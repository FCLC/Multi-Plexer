/*
 * Copyright (c) 2020 Yaroslav Pogrebnyak <yyyaroslav@gmail.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

extern "C" {

__global__ void Tonemap_Cuda_Reinhard(int x_position, int y_position, unsigned char* main, int main_linesize)
{
		
	
	//typical block and stride 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


	/*perform reinhart*/ 
	
	// create float to hold the initial pixel value
	float tone = main[x + y*main_linesize];
	// perform the mathematical shift 
	tone = tone* (tone /(tone+1));
	main[x + y*main_linesize] = tone;
}// end cuda function

__global__ void Tonemap_Cuda_Hable(int x_position, int y_position, unsigned char* main, int main_linesize)
{

    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;

        //typical block and stride 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


/*perform hable*/ 

        // create float to hold the initial pixel value
        float tone = main[x + y*main_linesize];
        // perform the mathematical shift 
        tone = tone*(A*tone+C*B)+D*E)/(tone*(A*tone+B)+D*F))-E/F

	// write the new value back to the frame
        main[x + y*main_linesize] = tone;


}//end cuda function

}//whole funtion

