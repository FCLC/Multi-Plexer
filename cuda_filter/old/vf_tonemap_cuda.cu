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



/*NOTICE: this is a test build based on the initial works of the NVIDIA Corporation to create an FF>
tonemapping filter. 
This filter will take in a source file that is presumed to be HDR (probably p010) 
and convert it to an aproximation of the source content within the SDR/ Rec.709 colour space 

Initially this will be done with the hable filter, as it is easier to implement and relatively simp>


Over time I hope to use the BT.2390-8 EOTF, but that is beyond the scope of the initial build
*/

/*
Changelog

2021/01/03
Creation of base files
2021/01/05

start from scratch- other approach seems silly 


*/


//This is the cuda side. it handles all of the math.

// let's start simple and work from the classic sum 2 arrays example and addapt it

//this funtion exists as it's on program/file. it handles everything when called upon by the relavant c file.


#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays


__global__
void add(float pix, int x, int y )// what do we need? we need a pointer to an array, call it source, and we need 2 integers, the width and the height
{

		// pix is a 2d array of width x and height y. therefore, to work on a given pixel, we adress it first by it's collum, then by it's row. 

		//to optimize this, we'll spin the threads into blocks of {[ceiling] x/32} width, which then get spun off into threads for each block. This can be optimized down the line


	/*
	this is the bit I need to wrap my head around, need to figure out how to translate it

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	*/


	//hable coefficients
	float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
		// might be better not to have them be variables, can remove redundant lookups or memory allocations
	for (int i = index; i < n; i += stride)
   	// here we modify and write back the value of the pixel
	 pix
}













int main(void) // for now it's a main, but it will be converted to a standard call right after.
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
