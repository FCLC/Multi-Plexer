/*
* Copyright (c) 2021 Felix LeClair 
*
*[I don't know if this is correct for a copyright notice, please correct me if wrong]
*
* Derived in part by the work of Nvidia in 2017 on the vf_thumbnail_cuda filter 
*
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



/*NOTICE: this is a test build based on the initial works of the NVIDIA Corporation to create an FFMPEG CUDA
tonemapping filter. 
This filter will take in a source file that is presumed to be HDR (probably p010) 
and convert it to an aproximation of the source content within the SDR/ Rec.709 colour space 

Initially this will be done with the hable filter, as it is easier to implement and relatively simple 


Over time I hope to use the BT.2390-8 EOTF, but that is beyond the scope of the initial build 
*/

/*
Changelog

2021/01/03 
Creation of base files 

*/

 
extern "C" {

//we need to return a float I believe, not sure how to return with a void without having to share memory across the 
//PCIE bus 

__global__ void Tonemap_linear(float in, float *x, float *y)
{

	//this spreads everything into blocks and threads
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  	//this is the loop that does the work 
	for (int i = index; i < n; i += stride)

	//this is where we do the math 
    static float hable(float in)
{
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}

}


}//last bracket


//Everything from the original function is here 


/*
__global__ void Thumbnail_uchar(cudaTextureObject_t uchar_tex,
                                int *histogram, int src_width, int src_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < src_height && x < src_width)
    {
        unsigned char pixel = tex2D<unsigned char>(uchar_tex, x, y);
        atomicAdd(&histogram[pixel], 1);
    }
}

__global__ void Thumbnail_uchar2(cudaTextureObject_t uchar2_tex,
                                 int *histogram, int src_width, int src_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < src_height && x < src_width)
    {
        uchar2 pixel = tex2D<uchar2>(uchar2_tex, x, y);
        atomicAdd(&histogram[pixel.x], 1);
        atomicAdd(&histogram[256 + pixel.y], 1);
    }
}

__global__ void Thumbnail_ushort(cudaTextureObject_t ushort_tex,
                                 int *histogram, int src_width, int src_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < src_height && x < src_width)
    {
        unsigned short pixel = (tex2D<unsigned short>(ushort_tex, x, y) + 128) >> 8;
        atomicAdd(&histogram[pixel], 1);
    }
}

__global__ void Thumbnail_ushort2(cudaTextureObject_t ushort2_tex,
                                  int *histogram, int src_width, int src_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < src_height && x < src_width)
    {
  
*/

