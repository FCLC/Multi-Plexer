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


/*
 * tonemapCUDAContext
 */
typedef struct tonemapCUDAContext {
    const AVClass      *class;

    enum AVPixelFormat in_format_tonemap;

    AVCUDADeviceContext *hwctx;

    CUcontext cu_ctx;
    CUmodule cu_module;
    CUfunction cu_func;
    CUstream cu_stream;

    FFFrameSync fs;

    int x_position;
    int y_position;

} tonemapCUDAContext;

/*
 * Helper to find out if provided format is supported by filter
 */
static int format_is_supported(const enum AVPixelFormat formats[], enum AVPixelFormat fmt)
{
    for (int i = 0; formats[i] != AV_PIX_FMT_NONE; i++)
        if (formats[i] == fmt)
            return 1;
    return 0;
}

/**
 * Call tonemaping kernell
 */
static int tonemap_cuda_call_kernel(
    tonemapCUDAContext *ctx,
    int x_position, int y_position,
    uint8_t* main_data, int main_linesize,
    int main_width, int main_height) {

    CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;

    void* kernel_args[] = {
        &x_position, &y_position,
        &main_data, &main_linesize,
       };

    return CHECK_CU(cu->cuLaunchKernel(
        ctx->cu_func,
        DIV_UP(main_width, BLOCK_X), DIV_UP(main_height, BLOCK_Y), 1,
        BLOCK_X, BLOCK_Y, 1,
        0, ctx->cu_stream, kernel_args, NULL));
}

/**
 * tonemap the darn thing
 */
static int tonemap_cuda_blend(FFFrameSync *fs)
{
    int ret;

    AVFilterContext *avctx = fs->parent;
    tonemapCUDAContext *ctx = avctx->priv;
    AVFilterLink *outlink = avctx->outputs[0];

    CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = ctx->hwctx->cuda_ctx;

    AVFrame *input_main;

    ctx->cu_ctx = cuda_ctx;

    // push cuda context

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0) {
        av_frame_free(&input_main);
        return ret;
    }

    // first plane

    tonemap_cuda_call_kernel(ctx, ctx->x_position, ctx->y_position, input_main->data[0], input_main->linesize[0], input_main->width, input_main->height);

    //  depending on pixel format

    switch(ctx->in_format_tonemap) {
    case AV_PIX_FMT_NV12:
        tonemap_cuda_call_kernel(ctx,
            ctx->x_position, ctx->y_position / 2,
            input_main->data[1], input_main->linesize[1],
            input_main->width, input_main->height / 2);
        break;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVA420P:
        tonemap_cuda_call_kernel(ctx,
            ctx->x_position / 2 , ctx->y_position / 2,
            input_main->data[1], input_main->linesize[1],
            input_main->width / 2, input_main->height / 2);

        tonemap_cuda_call_kernel(ctx,
            ctx->x_position / 2 , ctx->y_position / 2,
            input_main->data[2], input_main->linesize[2],
            input_main->width / 2, input_main->height / 2);
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "Passed unsupported pixel format\n");
        av_frame_free(&input_main);
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return AVERROR_BUG;
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    return ff_filter_frame(outlink, input_main);
}

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

/**
 * Activate tonemap_cuda
 */
static int tonemap_cuda_activate(AVFilterContext *avctx)
{
    tonemapCUDAContext *ctx = avctx->priv;

    return ff_framesync_activate(&ctx->fs);
}

/**
 * Query formats
 */
static int tonemap_cuda_query_formats(AVFilterContext *avctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };

    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(avctx, pix_fmts);
}

/**
 * Configure output
 */
static int tonemap_cuda_config_output(AVFilterLink *outlink)
{
	//definition of the cuda code after being turned to ptx. this then get's added as inline code to the c compiler output
	extern char vf_tonemap_cuda_ptx[];
       // extern char vf_tonemap_cuda_ptx[];
       // extern char vf_tonemap_cuda_ptx[];

    int err;
    AVFilterContext* avctx = outlink->src;
    tonemapCUDAContext* ctx = avctx->priv;

    AVFilterLink *inlink = avctx->inputs[0];
    AVHWFramesContext  *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;

    AVFilterLink *inlink_tonemap = avctx->inputs[1];
    AVHWFramesContext  *frames_ctx_tonemap = (AVHWFramesContext*)inlink_tonemap->hw_frames_ctx->data;

    CUcontext dummy, cuda_ctx;
    CudaFunctions *cu;

    // check input format

    if (!frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input, (did you remember to specify cuda?) \n");
        return AVERROR(EINVAL);
    }

    ctx->in_format_tonemap = frames_ctx->sw_format;
    if (!format_is_supported(supported_main_formats, ctx->in_format_tonemap)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported main input format: %s\n",
               av_get_pix_fmt_name(ctx->in_format_tonemap));
        return AVERROR(ENOSYS);
    }

    // initialize the hardware context 

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
/*
    err = CHECK_CU(cu->cuModuleLoadData(&ctx->cu_module, vf_tonemap_cuda_ptx));
    if (err < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return err;
    }
*/
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


static const AVOption tonemap_cuda_options[] = {
	//describes the options the filter provides. in this case we'll have hable, reinhart and clip(aka clamp)


 /*{ "x", "Overlay x position",
      OFFSET(x_position), AV_OPT_TYPE_INT, { .i64 = 0 }, INT_MIN, INT_MAX, .flags = FLAGS },
    { "y", "Overlay y position",
      OFFSET(y_position), AV_OPT_TYPE_INT, { .i64 = 0 }, INT_MIN, INT_MAX, .flags = FLAGS },
    { "eof_action", "Action to take when encountering EOF from secondary input ",
        OFFSET(fs.opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_REPEAT },
        EOF_ACTION_REPEAT, EOF_ACTION_PASS, .flags = FLAGS, "eof_action" },
        { "repeat", "Repeat the previous frame.",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, "eof_action" },
        { "endall", "End both streams.",            0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, "eof_action" },
        { "pass",   "Pass through the main input.", 0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, "eof_action" },
    { "shortest", "force termination when the shortest input terminates", OFFSET(fs.opt_shortest), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "repeatlast", "repeat overlay of the last overlay frame", OFFSET(fs.opt_repeatlast), AV_OPT_TYPE_BOOL, {.i64=1}, 0, 1, FLAGS },
    { NULL },*/
};

FRAMESYNC_DEFINE_CLASS(tonemap_cuda, tonemapCUDAContext, fs);

static const AVFilterPad tonemap_cuda_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
   /* {
        .name         = "overlay",
        .type         = AVMEDIA_TYPE_VIDEO,
    },*/
    { NULL }
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
    .description     = NULL_IF_CONFIG_SMALL("Tonemap a given video using a given algorithm via CUDA"),
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
