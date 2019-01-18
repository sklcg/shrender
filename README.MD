# Spherical rendering tool

A simple demo based on [Github:google/spherical harmonics](https://github.com/google/spherical-harmonics). 

## Dependencies:
+ Opencv >= 3.4.0
+ Eigen >= 3.3.4
+ Embree == 2.17.x
+ Assimp == 3.3

## Configuration
    
1. In CMakeLists.txt line 10: set the path to EIGEN 3
2. Use cmake-GUI to configure, and set the ASSIMP_DIR, OPENCV_DIR, EMBREE_DIR
3. Generate and compile, then you will get the shlight.exe

## Usages:

### Indispensable arguments

    shlight envrender  envmap  mesh              x y z  output
    shlight envrender+ envmap  mesh              x y z  output
    shlight  shrender  shfile  mesh                     output
    shlight  shrender+ shfile  mesh  background  x y z  output
    shlight eshrender  envmap  mesh              x y z  output
    shlight eshrender+ envmap  mesh              x y z  output
    shlight    env2sh  envmap                    x y z  output
    shlight    sh2env  shfile                           output
    shlight env2sh2env envmap                    x y z  output
    shlight   panorama envmap                    x y z  output
    shlight       view envmap                    x y z  output
    shlight       mask         mesh                     output
    shlight      shmap         mesh                     output

* set '+' for rendering the background, otherwise the background is black. *

### Optional arguments
    --spp      : Default(1024), Samples per pixel during rendering
    --fov      : Default(60.0), Field Of View (FOV) on horizontal (degree)
    --width    : Default( 256), Width of the output image
    --height   : Default( 256), Height of the output image
    --exposure : Default( 0.0), Scale the values of output image by 2^exposure
    --gamma    : Default( 2.2), The gamma curve applied to the output image
    --gfsize   : Default(   0), Kernal size of the gaussian filter applied to the output image.
    --pheight  : Default(   0), Only available in ‘shlight panorama’ 
                 Height of the output panorama, default 0, the same size as the input envmap

### For example:
    shlight envrender  0001.hdr horse.ply 0 0 -1 out.png
    shlight envrender+ 0001.hdr horse.ply 0 0 -1 out.exr
    shlight env2sh     0001.hdr           0 0 -1 sh1.txt
    shlight shrender    sh1.txt horse.ply        out.hdr
    shlight shmap     horse.ply              horse.shmap
    shlight panorama  

For getting the shmap in npy formmat, run:
    
    python3 convert.py xxx.shmap xxx.npy

