//---------------------------------------------------------------------------------------------------------------------
// mico-dnn
//---------------------------------------------------------------------------------------------------------------------
//  Copyright 2020 Ricardo Lopez Lopez (a.k.a. ricloplop) ricloplop@gmail.com & Pablo Ramon Soria (a.k.a. Bardo91) pabramsor@gmail.com
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


#ifndef MICO_FLOW_STREAMERS_BLOCKS_PROCESSORS_BLOCKDARKNET_H_
#define MICO_FLOW_STREAMERS_BLOCKS_PROCESSORS_BLOCKDARKNET_H_

#include <flow/Block.h>
#include <mico/slam/Dataframe.h>
#ifdef HAS_DARKNET
    #include <mico/dnn/object_detection/dnn/WrapperDarknet.h>
    #include <mico/dnn/map3d/Entity.h>
#endif
#include <opencv2/core/types.hpp>
#include <utility> 
namespace dnn{

    class BlockDarknet: public flow::Block{
    public:
        /// Get name of block
        virtual std::string name() const override {return "Darknet";}

        BlockDarknet();
        // ~BlockDarknet(){};

        bool configure(std::vector<flow::ConfigParameterDef> _params) override;
        /// Get list of parameters of the block
        std::vector<flow::ConfigParameterDef> parameters() override;


        /// Returns a brief description of the block
        std::string description() const override {return    "Block that implements darknet deep neuronal network for multiple 2D object detection.\n"
                                                            "   - Inputs: Confidence threshold: Minimun confidence treshold to detect an object \n"
                                                            "             Dense cloud: (true/false) Use dense cloud or feature cloud \n"
                                                            "             Radius removal: (true/false) Aply filter removal \n"
                                                            "             Radius search: (0.1) Radius in meters to search neighbors \n"
                                                            "             Minimum neighbors: (20) Minimun neighbors in a radius to be considered inliner \n"
                                                            "   - Outputs: \n"
                                                            "              Color Image: Color image with detected objects in a bounding box \n"
                                                            "              Entities: Vector of entities \n";};
    private:
        void objects_names_from_file(std::string const filename);
        cv::Scalar obj_id_to_color(int obj_id);

    private:
        bool idle_ = true;
        bool hasParameters_ = false; //weights, cfg, confidence threshold and use dense cloud
        
        float confidenceThreshold_ = 0.7;
        int numEntities_ = 0;
        bool useDenseCloud_ = false;
        bool storeClouds_ = false;
        // radius outlier removal parameters
        bool radiusFilter_ = false;
        double radiusSearch_ = 0.1;
        int minNeighbors_ = 20;

        // min cut filter parameters
        bool minCutFilter_ = false;
        double weightCutFilter_ = 0.004;
        double sigmaCutFilter_ = 0.005; 

        #ifdef HAS_DARKNET
        dnn::WrapperDarknet detector_;
        std::map<int, std::pair<std::string, cv::Scalar>> objNames_;
        #endif
    };

}

#endif