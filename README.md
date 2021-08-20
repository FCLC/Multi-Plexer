# Jetson_ffmpeg_trancode_cluster

 This project aims to bring together a few things together. 

The inciting goal was to:
lower my power consumptions
learn aarch64 (AKA 64 bit arm)
learn about clustering 
implement platform agnostic hardware accelerated transcoding for ffmpeg

I've also been tracking some ups and downs on the Level1Techs forum in this thread! https://forum.level1techs.com/t/building-a-10-100w-distributed-arm-based-transcode-cluster-i-missed-devember-jobless-january-floundering-february/167831/40


## Components 

### Hardware:

 1. Host server (either dedicated or worskstation PC)
 2. Networking switch (poe?)
 3. Multiple Nvidia Jetson's (ideally NX (agx is the dream)) but probably 2gb nano's or, depending on launch dates of Nvidia roadmap, Jetson Nano Next or Jetson Orin 

Other potential option: 

Turing Machines are launching a V2 of their Turring Pi cluster board, and it will have 4 260 pin slot's, initially deisgned for the Rpi CM4. I've reached out to their questions email adress asking about potential support, but have yet to hear back. Would alow for any mix of Nano (2 GB or 4GB) and Xavier NX + any other Jetson SOM that Nvidia may release. 
 
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
  2. client side of UT or KP from 1.2.
  3. custom ffmpeg build
   1. Personal cuda patch (should upstream to newest branch once I'm done) for tonemapping. Currently Reinhard and Hable, but will change to eotf BT.2390 Soon TM
   2. Jcover90 ffmpeg patch to enable the use of the transcode blocks 
   3. Nvidia build of ffmpeg to enable decoding and vf_scale_cuda
  4. Client side of load balancer from 1.6.
  5. (things I forgot will go here)
  
## How do I plan to architect this?
    


## Update August 2021

Been busy and burned out of late with other projects, and hoping to come pack to this one enventually. here's a layout for what needs to be done in a clear outline that isnt overwelming and can be takled in small bites.

Step 1: have ffmpeg tonemap using a filter 

Step 2: compile plex with custom filter built in 

step 3: have local plex call custom filter 

step 4: have local plex call external plex, without custom filtering 

step 5: have local plex call external plex, with custom filtering 

step 6: have local plex turn off unneeded nodes and turn them back on as needed. 

step 7: integration testing 

step 8: Enjoy power saving 