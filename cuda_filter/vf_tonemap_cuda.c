/*
* Copyright (c) 2021 Felix LeClair 
*
*[I don't know if this is correct for a copyright notice, please correct me if wrong]
*
* Derived in part by the work of Nvidia in 2017 on the vf_thumbnail_cuda filter and on the work of Yaroslav Pogrebnyak <yyyaroslav@gmail.com> on the vf_overlay_cuda filter

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

/*
Changelog

2021/01/03
Creation of base files
2021/01/05
start from scratch- other approach seems silly 
2021/01/05
RTFM and just get my shit together 

This is the C side.
All this file needs to do is:
Negotiate the filter 
get the frame 
get the information about the frame
pass the frame and information to the cuda side
receive the frame back 
send it on in the chain 
*/


/**
 * Initialize tonemap_cuda
 */
static av_cold int tonemap_cuda_init(AVFilterContext *avctx)
{
    tonemapCUDAContext* ctx = avctx->priv;
    ctx->fs.on_event = &tonemap_cuda_blend;

    return 0;
}

/**
 * Uninitialize tonemap_cuda
 */
static av_cold void tonemap_cuda_uninit(AVFilterContext *avctx)
{
    tonemapCUDAContext* ctx = avctx->priv;

    ff_framesync_uninit(&ctx->fs);

    if (ctx->hwctx && ctx->cu_module) {
        CUcontext dummy;
        CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;
        CHECK_CU(cu->cuCtxPushCurrent(ctx->cu_ctx));
        CHECK_CU(cu->cuModuleUnload(ctx->cu_module));
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }
}

//query_formats() goes here 
static int tonemap_cuda_query_formats(AVFilterContext *avctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };

    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(avctx, pix_fmts);
}



//Config_props() goes here 
static int tonemap_cuda_config_output(AVFilterLink *outlink)
{

    extern char vf_tonemap_cuda_ptx[];

    int err;
    AVFilterContext* avctx = outlink->src;
    tonemapCUDAContext* ctx = avctx->priv;

    AVFilterLink *inlink = avctx->inputs[0];
    AVHWFramesContext  *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;

    AVFilterLink *inlink_tonemap = avctx->inputs[1];
    AVHWFramesContext  *frames_ctx_tonemap = (AVHWFramesContext*)inlink_tonemap->hw_frames_ctx->data;

    CUcontext dummy, cuda_ctx;
    CudaFunctions *cu;

    // check main input formats

    if (!frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on main input\n");
        return AVERROR(EINVAL);
    }

    ctx->in_format_main = frames_ctx->sw_format;
    if (!format_is_supported(supported_main_formats, ctx->in_format_main)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported main input format: %s\n",
               av_get_pix_fmt_name(ctx->in_format_main));
        return AVERROR(ENOSYS);
    }

    // check tonemap input formats

    if (!frames_ctx_tonemap) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on tonemap input\n");
        return AVERROR(EINVAL);
    }

    ctx->in_format_tonemap = frames_ctx_tonemap->sw_format;
    if (!format_is_supported(supported_tonemap_formats, ctx->in_format_tonemap)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported tonemap input format: %s\n",
            av_get_pix_fmt_name(ctx->in_format_tonemap));
        return AVERROR(ENOSYS);
    }

    // check we can tonemap pictures with those pixel formats

    if (!formats_match(ctx->in_format_main, ctx->in_format_tonemap)) {
        av_log(ctx, AV_LOG_ERROR, "Can't tonemap %s on %s \n",
            av_get_pix_fmt_name(ctx->in_format_tonemap), av_get_pix_fmt_name(ctx->in_format_main));
        return AVERROR(EINVAL);
    }
 // initialize

    ctx->hwctx = frames_ctx->device_ctx->hwctx;
    cuda_ctx = ctx->hwctx->cuda_ctx;
    ctx->fs.time_base = inlink->time_base;

    ctx->cu_stream = ctx->hwctx->stream;

    outlink->hw_frames_ctx = av_buffer_ref(inlink->hw_frames_ctx);

    // load functions

    cu = ctx->hwctx->internal->cuda_dl;

    err = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (err < 0) {
        return err;
    }

    err = CHECK_CU(cu->cuModuleLoadData(&ctx->cu_module, vf_tonemap_cuda_ptx));
    if (err < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return err;
    }

    err = CHECK_CU(cu->cuModuleGetFunction(&ctx->cu_func, ctx->cu_module, "tonemap_Cuda"));
    if (err < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return err;
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    // init dual input

    err = ff_framesync_init_dualinput(&ctx->fs, avctx);
    if (err < 0) {
        return err;
    }

    return ff_framesync_configure(&ctx->fs);
}

//filer_frame() goes here 







/*NOTICE: this is a test build based on the initial works of the NVIDIA Corporation to create an FF>
tonemapping filter. 
This filter will take in a source file that is presumed to be HDR (probably p010) 
and convert it to an aproximation of the source content within the SDR/ Rec.709 colour space 

Initially this will be done with the hable filter, as it is easier to implement and relatively simp>


Over time I hope to use the BT.2390-8 EOTF, but that is beyond the scope of the initial build
*/





#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"

#include "avfilter.h"
#include "framesync.h"
#include "internal.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, ctx->hwctx->internal->cuda_dl, x)
#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

#define BLOCK_X 32
#define BLOCK_Y 16

static const enum AVPixelFormat supported_main_formats[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NONE,
};

static const enum AVPixelFormat supported_tonemap_formats[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUVA420P,
    AV_PIX_FMT_NONE,
};

/**
 * tonemapCUDAContext
 */
typedef struct tonemapCUDAContext {
    const AVClass      *class;

    enum AVPixelFormat in_format_tonemap;
    enum AVPixelFormat in_format_main;

    AVCUDADeviceContext *hwctx;

    CUcontext cu_ctx;
    CUmodule cu_module;
    CUfunction cu_func;
    CUstream cu_stream;

    FFFrameSync fs;

    int x_position;
    int y_position;

} tonemapCUDAContext;

/**
 * Helper to find out if provided format is supported by filter
 */
static int format_is_supported(const enum AVPixelFormat formats[], enum AVPixelFormat fmt)
{
    for (int i = 0; formats[i] != AV_PIX_FMT_NONE; i++)
            return 1;
    return 0;
}

/**
 * Helper checks if we can process main and tonemap pixel formats
 */
static int formats_match(const enum AVPixelFormat format_main, const enum AVPixelFormat format_tonemap) {
    switch(format_main) {
    case AV_PIX_FMT_NV12:
        return format_tonemap == AV_PIX_FMT_NV12;
    case AV_PIX_FMT_YUV420P:
        return format_tonemap == AV_PIX_FMT_YUV420P ||
               format_tonemap == AV_PIX_FMT_YUVA420P;
    default:
        return 0;
    }
}








//Standard ffmpegs options for documentation



static const AVFilterPad tonemap_cuda_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
};

static const AVFilterPad tonemap_cuda_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = &tonemap_cuda_config_output,
    },
    { NULL }
};

AVFilter ff_vf_tonemap_cuda = {
    .name            = "tonemap_cuda",
    .description     = NULL_IF_CONFIG_SMALL("tonemap video using CUDA"),
    .priv_size       = sizeof(tonemapCUDAContext),
    .priv_class      = &tonemap_cuda_class,
    .init            = &tonemap_cuda_init,
    .uninit          = &tonemap_cuda_uninit,
    .activate        = &tonemap_cuda_activate,
    .query_formats   = &tonemap_cuda_query_formats,
    .inputs          = tonemap_cuda_inputs,
    .outputs         = tonemap_cuda_outputs,
    .preinit         = tonemap_cuda_framesync_preinit,
    .flags_internal  = FF_FILTER_FLAG_HWFRAME_AWARE,
};
