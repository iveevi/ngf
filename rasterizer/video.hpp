extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>
}

struct VideoWriter {
	struct AVCodec         *codec;
	struct AVCodecContext  *codec_context;
	struct AVFormatContext *format_context;
	struct AVFrame         *frame;
	struct AVStream        *stream;
	struct SwsContext      *sws_context;
	uint32_t               frame_index;

	// MP4 format
};
