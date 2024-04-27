#ifndef _RKNN_MODEL_ZOO_COMMON_H_
#define _RKNN_MODEL_ZOO_COMMON_H_

#include "rknn_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     80
#define NMS_THRESH        0.45
#define BOX_THRESH        0.25
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

typedef struct {
    unsigned char b {0};
    unsigned char g {0};
    unsigned char r {0};
} BGR;

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    int input_image_width;
    int input_image_height;
    bool is_quant;
} rknn_app_context_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct
{
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct
{
    uint8_t *seg_mask;
} object_segment_result;

typedef struct
{
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
    object_segment_result results_seg[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

inline int clamp(int   val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
inline int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

void dump_tensor_attr(rknn_tensor_attr* attr);
unsigned char* load_model(const char* filename, int& fileSize);

#endif //_RKNN_MODEL_ZOO_COMMON_H_
