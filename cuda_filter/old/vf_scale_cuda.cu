/*
 * original source Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Change to tonemap style filter copyright Felix LeClair 
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*

Warning: this is a VERY early alpha of a cuda accelerated filter to tonemap. Please see ffmpeg devel mailing list for message of title [vf_tonemap_cuda] VERY alpha ground work- implemented as cuda frame
sent on the 14th of January 2021 
It's poorly written and documented. this should not be merged under any circumstance in it's present form.


*/


#include "cuda/vector_helpers.cuh"

template<typename T>
__device__ inline void Subsample_Nearest(cudaTextureObject_t tex,
                                         T *dst,
                                         int dst_width, int dst_height, int dst_pitch,
                                         int src_width, int src_height,
                                         int bit_depth)
/*
tex is the cuda texture
T is a pointer to the destination frame
dst_width is the width of the output frame
dst_height is the height of the output frame 
dst_pitch is the I DON'T KNOW YET, but I suspect this has to do when changing the size of pixels when shifting aspect ratios.
	 as such I'm going to redifine as 1 so I don't have any issues 
bit_depth  is the amount of bits per colour channel
*/

{
	
	dst_pitch =1;// this is a bodge, but won't be needed when I change the rest of the source to not need to deal with the legacy scalling source code.
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (yo < dst_height && xo < dst_width)
    {
        float hscale = (float)src_width / (float)dst_width;// supposed to be the scalling factor in the original funtion, but I'm going to ignore it
        float vscale = (float)src_height / (float)dst_height; // as above, going to ignore it
        float xi = (xo + 0.5f); // * hscale;
        float yi = (yo + 0.5f); // * vscale;
	float val_IN = tex2D<T>(tex, xi, yi);// to start I'm doing reinhard because it's idiot proof
	float out = val_IN*(val_IN/(val_IN + 1.0f)); // this scales the incoming pixel by a factor of x/(x+1). this guarentees a value between 0 and 1. far from the best algortihm, but is fit for purpose 
	dst[yo*dst_pitch+xo] =out; // this is where I'm transforming the value to the tonemapped value.  
    }
}


extern "C" {

#define NEAREST_KERNEL(T) \
    __global__ void Subsample_Nearest_ ## T(cudaTextureObject_t src_tex,                  \
                                            T *dst,                                       \
                                            int dst_width, int dst_height, int dst_pitch, \
                                            int src_width, int src_height,                \
                                            int bit_depth)                                \
    {                                                                                     \
	//call the device side  code under __device__ inline void Subsample_Nearest
        Subsample_Nearest<T>(src_tex, dst,                                                \
                              dst_width, dst_height, dst_pitch,                           \
                              src_width, src_height,                                      \
                              bit_depth);                                                 \
    }

NEAREST_KERNEL(uchar)
NEAREST_KERNEL(uchar2)
NEAREST_KERNEL(uchar4)

NEAREST_KERNEL(ushort)
NEAREST_KERNEL(ushort2)
NEAREST_KERNEL(ushort4)
}
