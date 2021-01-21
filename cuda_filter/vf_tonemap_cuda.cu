/* 

Copyright (c) 2021 Felix LeClair <felix.leclair123@hotmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

 */

/*

Doxy:

TLDR; tonemap a given video using a given tone curve via CUDA hardware acceleration

The primary purpose of tonemapping has to do with how HDR (High Dynamic Range) video content is distributed.
In essence, HDR content does not hard-code the real colour values of it's pixels. Instead it provides a linear
map of colours with very little natural contrast. Metadata is then used to transform the content to appear as
the creator intended, regardless of the capabilities of the users display. This allows for content to be consistent
across a diverse set of installations and provide a known minimum level of quality. Unfortunately, when transcoding
content, this metadata can rarely be used in the new file. As such the output tends to be extremely dim
and desaturated. This is reminiscent of LOG footage in digital cinema footage at time of capture.

As such, as part of the chain, when using HDR source and transcoding to SDR (Standard Dynamic Range) a tone curve
needs to be applied to mimic the effect that the metadata would have done when sent to the display. Opinions and
standards differ on the best approach for tonemapping content. As such this filter will include multiple options.

further reading on different standards and their approximate implementations in c++ can be found
at: https://web.archive.org/web/20201101024948/https://64.github.io/tonemapping/
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

__global__ void Tonemap_Cuda(int x_position, int y_position, unsigned char* main, int main_linesize)
{	/*default case is clip*/

        //typical block and stride 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

        // create float to hold the initial pixel value
        float tone = main[x + y*main_linesize];
        // perform the mathematical shift 
        tone = tone>>2; //10 bit to 8 bit, truncate least significant bits 
        main[x + y*main_linesize] = tone;
}// end cuda function




}//whole funtion

