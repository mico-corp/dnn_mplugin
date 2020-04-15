//---------------------------------------------------------------------------------------------------------------------
// mico-dnn 
//---------------------------------------------------------------------------------------------------------------------
//  Copyright 2020 - Ricardo Lopez Lopez (a.k.a. ricloplop) ricloplop@gmail.com 
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

#include <flow/flow.h>

#include <mico/dnn/flow/BlockDarknet.h>
#include <mico/dnn/flow/BlockEntityDatabase.h>
#include <mico/dnn/flow/BlockEntityOptimizer.h>
#include <mico/dnn/flow/BlockEntityInspector.h>

FLOW_TYPE_REGISTER(entity, std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>)
FLOW_TYPE_REGISTER(v_entity, std::vector<std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>>)

namespace dnn{
    extern "C" flow::PluginNodeCreator* factory(){
            flow::PluginNodeCreator *creator = new flow::PluginNodeCreator;

            creator->registerNodeCreator([](){ return std::make_unique<flow::FlowVisualBlock<dnn::BlockDarknet>>();         }, "DNN");
            creator->registerNodeCreator([](){ return std::make_unique<flow::FlowVisualBlock<dnn::BlockEntityDatabase>>();  }, "DNN");
            creator->registerNodeCreator([](){ return std::make_unique<flow::FlowVisualBlock<dnn::BlockEntityOptimizer>>(); }, "DNN");
            creator->registerNodeCreator([](){ return std::make_unique<flow::FlowVisualBlock<dnn::BlockEntityInspector>>(); }, "DNN");

            return creator;
        }
}