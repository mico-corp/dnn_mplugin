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


#ifndef MICO_FLOW_STREAMERS_BLOCKS_BLOCKENTITYOPTIMIZER_H_
#define MICO_FLOW_STREAMERS_BLOCKS_BLOCKENTITYOPTIMIZER_H_

#include <flow/Block.h>
#include <mico/slam/BundleAdjuster_g2o.h>

#include <mico/slam/cjson/json.h>
#ifdef HAS_DARKNET
    #include <mico/dnn/map3d/Entity.h>
#endif

namespace dnn{

    class BlockEntityOptimizer: public flow::Block{
    public:
        virtual std::string name() const override {return "Entity Optimizer";}

        BlockEntityOptimizer();
        // ~BlockEntityOptimizer();
    
        bool configure(std::unordered_map<std::string, std::string> _params) override;
        std::vector<std::string> parameters() override;        

        std::string description() const override {return    "Block that implements a semantic optimizer.\n"
                                                            "   - Inputs: \n"
                                                            "   - Outputs: \n";};
    
    private:
        mico::BundleAdjuster_g2o<pcl::PointXYZRGBNormal> optimizer_;
        bool idle_ = true;
        
    };

}

#endif