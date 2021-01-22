# Jetson_ffmpeg_trancode_cluster

 This project aims to bring together a few things together. 

The inciting goal was to:
lower my power consumptions
learn aarch64 (AKA 64 bit arm)
learn about clustering 
implement platform agnostic hardware accelerated transcoding for ffmpeg

## Components 

### Hardware:

 1. Host server (either dedicated or worskstation PC)
 2. Networking switch (poe?)
 3. Multiple Nvidia Jetson's (ideally NX (agx is the dream)) but probably 2gb nano's or, depending on launch dates of Nvidia roadmap, Jetson Nano Next or Jetson Orin 
 
 ### Software: 
 1. Host 
  1. OS for server- debian based 
  2. Unicorn Transcoder or Kuber Plex 
  3. NFS share (Data itself is accessed over the network on my nas, ZFS and so on)
  4. Custom capture scrip to modify arguments sent from PMS to transcoder 
  5. Plex Media Server
  6. Load balancer  
 
 2. Jetson (aka transcoder node)
  1. Jetpack (4.3?)
  2. client side of UT or KP from 1.b
  3. custom ffmpeg build
   1. Personal cuda patch (should upstream to newest branch once I'm done) for tonemapping. currently reihard, but will change to eotf 2390 eventually
   2. Jcover90 ffmpeg patch to enable the use of the transcode blocks 
   3. nvidia build of ffmpeg to enable decoding and vf_scale_cuda
  4. client side of load balancer from 1.f
  5. (things I forgot will go here)
  
## How do I plan to architect this?

As of now (dec 2020) the intention is to handle the different aspects using these projects:

The distribution of work load via the Unicorn transcoder project [link goes here]

Transcoding of content handled via ffmpeg [link to website here]

Jetson NANO does not currently have a great FFMPEG implementation for what I need. need to patch in 

As of now the most optimal way to achieve what we want is to decode in hardware, use cuda to resize and tonemap if needed, then encode the file while sending back to the host.      
    
## Current State of different parts

cuda based tonemapping filter: Pretty much done. Need to confirm performance (Don't have a jetson yet, test on old gtx670 as POC). GTX670 has 2gb of ram, so If I can simulate only having 768mb of vram, should be worse case scenario as POC

Capture Script to modify arguments to HW counterparts: Haven't begun developing; have spoken with UT devs about where to capture arguments. it's node JS, so will have to learn that

Test "cost" of pulling data across network, and optane on Jetson.  
