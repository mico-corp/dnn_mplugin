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


#ifndef MICO_FLOW_STREAMERS_BLOCKS_BLOCKENTITYDATABASE_H_
#define MICO_FLOW_STREAMERS_BLOCKS_BLOCKENTITYDATABASE_H_

#include <flow/Block.h>
#include <mico/slam/cjson/json.h>
#ifdef HAS_DARKNET
    #include <mico/dnn/map3d/Entity.h>
#endif

namespace dnn{

    class BlockEntityDatabase: public flow::Block{
    public:
        virtual std::string name() const override {return "Entity Database";}

        BlockEntityDatabase();
        // ~BlockEntityDatabase();
    
        bool configure(std::unordered_map<std::string, std::string> _params) override;
        std::vector<std::string> parameters() override;        

        std::string description() const override {return    "Block that implements a semantic database.\n"
                                                            "   - Inputs: \n"
                                                            "   - Outputs: \n";};
    private:
        void reinforceEntity(std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>);
    private:
        #ifdef HAS_DARKNET
            // [label][vector of entities]
            std::map<int, std::vector<std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>>> entities_;
        #endif
        float overlapScore_ = 1;
        int comparedEntities_ = 8;
        bool hasPrev_ = false;
        bool idle_ = true;
    };

}

#endif