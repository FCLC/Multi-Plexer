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



//This is the C side.
// All this file needs to do is call the relevant .cu file, 
// accept the output back and send it on in the FFmpeg pipe.
