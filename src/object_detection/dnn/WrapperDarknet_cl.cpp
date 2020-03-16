//---------------------------------------------------------------------------------------------------------------------
// mico-dnn
//---------------------------------------------------------------------------------------------------------------------
//  Copyright 2018 Pablo Ramon Soria (a.k.a. Bardo91) pabramsor@gmail.com
//---------------------------------------------------------------------------------------------------------------------
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//  and associated documentation files (the "Software"), to deal in the Software without restriction,
//  including without limitation the rights to use, copy, modify, merge, publish, distribute,
//  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all copies or substantial
//  portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
//  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
//  OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//---------------------------------------------------------------------------------------------------------------------

#include <mico/dnn/object_detection/dnn/WrapperDarknet_cl.h>
#include <chrono>


namespace dnn {
    bool WrapperDarknet_cl::init(std::string mModelFile, std::string mWeightsFile){
	#ifdef HAS_DARKNET_CL

        cl_set_device(0);

        mNet = load_network(const_cast<char*>(mModelFile.c_str()), 
                            const_cast<char*>(mWeightsFile.c_str()), 
                            0);
        set_batch_network(mNet, 1);
        srand(2222222);

        return mNet != nullptr;
	#else
	    return false;
	#endif
    }

    std::vector<std::vector<float>> WrapperDarknet_cl::detect(const cv::Mat &_img) {
	#ifdef HAS_DARKNET_CL
        if(mNet == nullptr || _img.rows == 0){
            return std::vector<std::vector<float>>();
        }

        // auto t1 = std::chrono::high_resolution_clock::now();        
        // Create container
        layer inputLayer = mNet->layers[0];
        float h = inputLayer.h;//iplImg->height;
        float w = inputLayer.w;//iplImg->width;
        float c = inputLayer.c;//iplImg->nChannels;
        if(mLastW != w || mLastH != h || mLastC != c){
            mLastW = w; mLastH = h; mLastC = c;
            mImage = make_image(w, h, c);
        }
        // auto t2 = std::chrono::high_resolution_clock::now();

        float stepX = _img.cols/w;
        float stepY = _img.rows/h;
        float stepC = _img.channels()/c;
        int counterI = 0;
        //#pragma omp parallel for
        for (float i = 0; i < _img.rows; i+=stepY) {
            int counterJ = 0;
            for (float j = 0; j < _img.cols; j+=stepX) {
                int counterC = 2;
                for (float k = 0; k < _img.channels(); k+=stepC) {
                    mImage.data[int(counterC * w * h + counterI * w + counterJ)] = _img.at<cv::Vec3b>(cv::Point(j, i))[k] / 255.;
                    counterC--;
                }
                counterJ++;
            }
            counterI++;
        }

        layer l = mNet->layers[mNet->n - 1];

        // auto t5 = std::chrono::high_resolution_clock::now();
        float *X = mImage.data;
        network_predict(mNet, X);
        // auto t6 = std::chrono::high_resolution_clock::now();
        int nboxes = 0;
        detection *dets = get_network_boxes(mNet, mImage.w, mImage.h, thresh, hier_thresh, 0, 1, &nboxes);
        // auto t7 = std::chrono::high_resolution_clock::now();
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        // auto t8 = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<float>> result;
        for (int i = 0; i < nboxes; ++i) {
            int classId = -1;
            float prob = 0;
            for (int j = 0; j < l.classes; ++j) {
                //if (dets[i].prob[j] > thresh) {
                    if (dets[i].prob[j] > prob) {
                        classId = j;
                        prob = dets[i].prob[j];
                    }
                    // printf("%d: %.0f%%\n", classId, dets[i].prob[j]*100);
                //}
            }

            if (classId >= 0 && prob > thresh) {
                int width = mImage.h * .006;

                box b = dets[i].bbox;
                //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                int left = (b.x - b.w / 2.) * _img.cols;
                int right = (b.x + b.w / 2.) * _img.cols;
                int top = (b.y - b.h / 2.) * _img.rows;
                int bot = (b.y + b.h / 2.) * _img.rows;

                if (left < 0)
                    left = 0;
                if (right > _img.cols - 1)
                    right = _img.cols - 1;
                if (top < 0)
                    top = 0;
                if (bot > _img.rows - 1)
                    bot = _img.rows - 1;

                result.push_back({classId, prob, left, top, right, bot});
            }
        }
        free_detections(dets, nboxes);
        // auto t9 = std::chrono::high_resolution_clock::now();

        // std::cout << "YOLO: image prep: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << std::endl; 
        // std::cout << "YOLO: image copy: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << std::endl; 
        // std::cout << "YOLO: letter box: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << std::endl; 
        // std::cout << "YOLO: data init : " << std::chrono::duration_cast<std::chrono::milliseconds>(t5-t4).count() << std::endl; 
        // std::cout << "YOLO: predict   : " << std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count() << std::endl; 
        // std::cout << "YOLO: get boxes : " << std::chrono::duration_cast<std::chrono::milliseconds>(t7-t6).count() << std::endl; 
        // std::cout << "YOLO: nms       : " << std::chrono::duration_cast<std::chrono::milliseconds>(t8-t7).count() << std::endl; 
        // std::cout << "YOLO: get result: " << std::chrono::duration_cast<std::chrono::milliseconds>(t8-t7).count() << std::endl; 

        return result;
	#else
	    return 	std::vector<std::vector<float>>();
	#endif
    }
}
