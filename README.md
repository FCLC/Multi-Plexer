# Jetson_ffmpeg_trancode_cluster

 This project aims to bring together a few things together. 

The inciting goal was to:
lower my power consumptions
learn aarch64 (AKA 64 bit arm)
learn about clustering 
implement platform agnostic hardware accelerated transcoding for ffmpeg


## How do I plan to architect this?

As of now (dec 2020) the intention is to handle the different aspects using these projects:

The distribution of work load via the Unicorn transcoder project [link goes here]

Transcoding of content handled via ffmpeg [link to website here]

Jetson NANO does not currently have a great FFMPEG implementation to leverage the hardware blocks.

As of now the most optimal way to achieve what we want is to decode in hardware, use openCL to resize and tonemap if needed, then encode the file while sending back to the host 

OpenCL is not currently supported, and the Nvidia package of FFmpeg does not support encoding. Therefore, the plan is to take advantage of the POCL translation interface to use openCL filters running on cuda. In adition to this, there is a community plugin that allows for both encoding and decoding through a common interface
     
    
