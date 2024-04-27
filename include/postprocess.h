#ifndef _RKNN_YOLOV8_SEG_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV8_SEG_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rk_common.h"

int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h, object_detect_result_list *od_results);


#endif //_RKNN_YOLOV8_SEG_DEMO_POSTPROCESS_H_
