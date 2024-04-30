// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Modified by Q-engineering 4-6-2024
//

/*-------------------------------------------
                Includes
-------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cstdlib>                  // for malloc and free
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "yolov8_seg.h"
#include "postprocess.h"
#include "rknn_api.h"
#include "rk_common.h"

static const char* labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
    float f;
    float FPS[16];
    int i, Fcnt=0;
    std::chrono::steady_clock::time_point Tbegin, Tend;
    char*          model_name = NULL;
    const char*    imagepath = argv[1];
    int            ret;

    unsigned char class_colors[][3] = {
        {255, 56, 56},   // 'FF3838'
        {255, 157, 151}, // 'FF9D97'
        {255, 112, 31},  // 'FF701F'
        {255, 178, 29},  // 'FFB21D'
        {207, 210, 49},  // 'CFD231'
        {72, 249, 10},   // '48F90A'
        {146, 204, 23},  // '92CC17'
        {61, 219, 134},  // '3DDB86'
        {26, 147, 52},   // '1A9334'
        {0, 212, 187},   // '00D4BB'
        {44, 153, 168},  // '2C99A8'
        {0, 194, 255},   // '00C2FF'
        {52, 69, 147},   // '344593'
        {100, 115, 255}, // '6473FF'
        {0, 24, 236},    // '0018EC'
        {132, 56, 255},  // '8438FF'
        {82, 0, 133},    // '520085'
        {203, 56, 255},  // 'CB38FF'
        {255, 149, 200}, // 'FF95C8'
        {255, 55, 199}   // 'FF37C7'
    };

    for(i=0;i<16;i++) FPS[i]=0.0;

    if (argc < 3) {
        fprintf(stderr,"Usage: %s [imagepath] [model]\n", argv[0]);
        return -1;
    }
    model_name = argv[2];

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_yolov8_seg_model(model_name, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_seg_model fail! ret=%d model_name=%s\n", ret, model_name);
        return -1;
    }

    while(1){
        cv::Mat orig_img;
        orig_img=cv::imread(imagepath, 1);
        if(orig_img.empty()) {
            printf("Error grabbing\n");
            break;
        }

        Tbegin = std::chrono::steady_clock::now();

        object_detect_result_list od_results;

        inference_yolov8_seg_model(&rknn_app_ctx, &orig_img, &od_results);

        // draw mask
        if (od_results.count >= 1){
            int wd = orig_img.cols;
            int ht = orig_img.rows;
            uint8_t *seg_mask = od_results.results_seg[0].seg_mask;
            for(int y = 0; y < ht; y++){
                uint8_t *seg_mask_h = &(seg_mask[y*wd]);
                for(int x = 0; x < wd; x++){
                    if (seg_mask_h[x] != 0){
                        int co = seg_mask_h[x] % 20;
                        BGR &bgr = orig_img.ptr<BGR>(y)[x];
                        bgr.b = cv::saturate_cast<uchar>(bgr.b * 0.5 + class_colors[co][0] * 0.5);
                        bgr.g = cv::saturate_cast<uchar>(bgr.g * 0.5 + class_colors[co][1] * 0.5);
                        bgr.r = cv::saturate_cast<uchar>(bgr.r * 0.5 + class_colors[co][2] * 0.5);
                    }
                }
            }
        }

        // draw objects
        char text[256];
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det_result = &(od_results.results[i]);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            cv::rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2),cv::Scalar(255, 0, 0));

        //            printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
        //                   det_result->box.right, det_result->box.bottom, det_result->prop);

            //put some text
            sprintf(text, "%s %.1f%%", labels[clamp(det_result->cls_id,0,80)], det_result->prop * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = det_result->box.left;
            int y = det_result->box.top - label_size.height - baseLine;
            if (y < 0) y = 0;
            if (x + label_size.width > orig_img.cols) x = orig_img.cols - label_size.width;

            cv::rectangle(orig_img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

            cv::putText(orig_img, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        Tend = std::chrono::steady_clock::now();
        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(orig_img, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

            //show output
//        std::cout << "FPS" << f/16 << std::endl;
        imshow("Radxa zero 3W - 1,8 GHz - 4 Mb RAM", orig_img);
        char esc = cv::waitKey(2);
        if(esc == 27) break;

//      imwrite("./out.jpg", orig_img);
    }

    release_yolov8_seg_model(&rknn_app_ctx);

    return 0;
}
