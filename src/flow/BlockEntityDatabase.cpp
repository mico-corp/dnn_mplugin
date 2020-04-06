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

#include <mico/dnn/flow/BlockEntityDatabase.h>
#include <flow/Policy.h>
#include <flow/Outpipe.h>
#include <flow/DataFlow.h>
#include <mico/slam/utils2d.h>
#include <sstream>

namespace dnn{

    BlockEntityDatabase::BlockEntityDatabase(){ 
        createPipe("Entities", "v_entity");
        
        createPolicy({{"Entities", "v_entity"}});

        registerCallback({"Entities"}, 
                                [&](flow::DataFlow _data){
                                    if(idle_){
                                        idle_ = false;
                                        #ifdef HAS_DARKNET
                                        auto entities = _data.get<std::vector<std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>>>("Entities"); 
                                        // store the new entities
                                        std::vector<std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>> newEntities;
                                        if(entities.size()> 0)
                                            std::cout << "[BlockEntityDatabase] Started entity database with first entity id " << entities[0]->id() << std::endl; 
                                        if(!entities_.empty()){
                                            // candidates
                                            for(auto queryE: entities){
                                                int label = queryE->label();
                                                // check overlap

                                                bool newEntity = true;
                                                std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>> parentEntity = nullptr;
                                                float affinity = 0;
                                                auto t1 = std::chrono::high_resolution_clock::now();
                                                // entities of same label in database
                                                int i = 0;
                                                for(auto trainE = entities_[label].rbegin(); trainE != entities_[label].rend() && i < comparedEntities_; ++trainE, i++){
                                                    if(queryE->id() != (*trainE)->id()){
                                                        float overlaped = queryE->percentageOverlapped(*trainE);
                                                        std::cout << "[BlockEntityDatabase] Overlapped percentage between " << queryE->id() << " and " << (*trainE)->id() << " : " << 
                                                            overlaped << std::endl;

                                                        // if the entity overlaps with other created dont create a new one and update the first
                                                        if(overlaped > overlapScore_ && overlaped > affinity){
                                                            parentEntity = (*trainE);
                                                            affinity = overlaped;
                                                            newEntity = false;
                                                        }
                                                    }   
                                                }

                                                auto t2 = std::chrono::high_resolution_clock::now();
                                                std::chrono::duration<float,std::milli> overlapTime = (t2 - t1);
                                                std::cout << "[BlockEntityDatabase]Time checking overlaping between cubes: " << overlapTime.count()/1000 << std::endl;
                                                // create new entity associated to the most related parent
                                                if(newEntity){
                                                    entities_[label].push_back(queryE);
                                                    newEntities.push_back(queryE);
                                                    std::cout << "[BlockEntityDatabase] Created new entity " << queryE->id() << "(" << queryE->name() << ")" << std::endl;
                                                }
                                                else{
                                                    // update entity 
                                                    parentEntity->update(queryE);
                                                    parentEntity->createWords();
                                                    std::cout << "[BlockEntityDatabase] Updated entity " << parentEntity->id() << "(" << parentEntity->name() << ")" << "  overlaped% " << affinity << " and creating words" << std::endl;
                                                }
                                            }
                                        }else{
                                            for(auto e: entities){
                                                entities_[e->label()].push_back(e);
                                                std::cout << "[BlockEntityDatabase] Added entity " << e->id() << "(" << e->name() << ")" << " to database" << std::endl; 
                                                newEntities.push_back(e);
                                                // check overlapping here maybe
                                            }
                                        }
                                        for(auto &label: entities_){
                                            for(auto &e: label.second){
                                                std::cout << "[BlockEntityDatabase] Entity: " << e->id() <<  " (" << e->name() << ") has appeared in " << e->dfs().size() << " dataframes and contains " 
                                                << e->words().size() << " words" << std::endl;
                                            
                                            }
                                        }
                                        getPipe("Entities")->flush(newEntities);
                                        idle_ = true;
                                        #endif
                                    }
                                }
        );
    }

    // BlockEntityDatabase::~BlockEntityDatabase(){
    // } 

    bool reinforceEntity(std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>> _e){
        if(_e->dfs().size()>1){
            auto dfs = _e->dfMap();
            auto multimatchesinliers = _e->crossReferencedInliers();
            auto createdWords = _e->computedWords();
            int newMatches = 0;

            // compare features from all detections 
            for(auto &querydf: dfs){
                for(auto &traindf: dfs){
                    auto queryId = querydf.second->id();
                    auto trainId = traindf.second->id();

                    // avoid compare with himself
                    if(queryId == trainId)
                        continue;

                    // check if the matches are already computed
                    if((createdWords.find(std::make_pair(queryId, trainId)) == createdWords.end()) && (createdWords.find(std::make_pair(trainId, queryId)) == createdWords.end())){
                        // compute the match
                        std::vector<cv::DMatch> matches;
                        if (!mico::matchDescriptorsBF(_e->descriptors(queryId), _e->descriptors(trainId), matches, 1, 8))
                                continue;

                        // add mmi
                        _e->crossReferencedInliers(queryId, trainId, matches);
                        newMatches++;
                    }
                }
            }
            if(newMatches > 0){
                _e->createWords();
                return true;
            }
        }
        return false;
    }


    bool BlockEntityDatabase::configure(std::unordered_map<std::string, std::string> _params){
        cjson::Json jParams;
        for(auto &param: _params){
            if(param.second == "")
                return false;
            if(param.first == "overlapScore"){
                std::istringstream istr(_params["overlapScore"]);
                istr >> overlapScore_;
                jParams["overlapScore"] = overlapScore_;
            }
            if(param.first == "compared_entities"){
                std::istringstream istr(_params["compared_entities"]);
                istr >> comparedEntities_;
                jParams["compared_entities"] = comparedEntities_;
            }
            
        }
        std::cout << "[BlockEntityDatabase]Score selected: " << overlapScore_ << std::endl;
        std::cout << "[BlockEntityDatabase]Entities compared: " << comparedEntities_ << std::endl;
    }
    
    std::vector<std::string> BlockEntityDatabase::parameters(){
        return {"overlapScore","compared_entities"};
    }
}
