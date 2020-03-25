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

#include <mico/dnn/flow/BlockDarknet.h>
#include <flow/Policy.h>
#include <flow/Outpipe.h>
#include <flow/DataFlow.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <chrono>
#include <iostream>
#include <experimental/filesystem>
#include <pcl/common/geometry.h>
namespace dnn{

    BlockDarknet::BlockDarknet(){

        createPipe("Color Image", "image");
        createPipe("Entities", "v_entity");

        createPolicy({ {"Color Image", "image"}, 
                        {"Dataframe", "dataframe"}});

        registerCallback({"Color Image"}, 
                                [&](flow::DataFlow _data){
                                    if(idle_){
                                        idle_ = false;
                                        #ifdef HAS_DARKNET
                                        if(hasParameters_){
                                            cv::Mat image;
                                            // check data received
                                            try{
                                                image = _data.get<cv::Mat>("Color Image").clone();
                                            }catch(std::exception& e){
                                                std::cout << "Failure Darknet. " <<  e.what() << std::endl;
                                                idle_ = true;
                                                return;
                                            }
                                            // vector of detected entities 
                                            std::vector<std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>> entities;

                                            // get image detections
                                            auto detections = detector_.detect(image);
                                            // detection -> label, confidence, left, top, right, bottom
                                            for(auto &detection: detections){
                                                // confidence threshold 
                                                if(detection[1]>confidenceThreshold_){
                                                    std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>> e(new dnn::Entity<pcl::PointXYZRGBNormal>(
                                                         numEntities_, detection[0], detection[1], {detection[2],detection[3],detection[4],detection[5]}));                                                                                          
                                                    entities.push_back(e);
                                                    numEntities_++;
                                                    cv::Rect rec(detection[2], detection[3], detection[4] -detection[2], detection[5]-detection[3]);
                                                    //cv::putText(image, "Confidence" + std::to_string(detection[1]), cv::Point2i(detection[2], detection[3]),1,2,cv::Scalar(0,255,0));
                                                    cv::putText(image, "ObjectId: " + std::to_string(detection[0]), cv::Point2i(detection[2], detection[3]),1,2,cv::Scalar(0,255,0));
                                                    cv::rectangle(image, rec, cv::Scalar(0,255,0));
                                                }
                                            }

                                            // send image with detections
                                            if(getPipe("Color Image")->registrations() !=0 )
                                                getPipe("Color Image")->flush(image);
                                            // send entities
                                            if(entities.size()>0 && getPipe("Entities")->registrations() !=0 )
                                                getPipe("Entities")->flush(entities);
                                            
                                        }else{
                                            std::cout << "No weights and cfg provided to Darknet\n";
                                        }
                                        #endif
                                        idle_ = true;
                                    }
                                });

        registerCallback({"Dataframe"}, 
                                [&](flow::DataFlow _data){
                                    if(idle_){
                                        idle_ = false;
                                        #ifdef HAS_DARKNET
                                        if(hasParameters_){
                                            cv::Mat image;
                                            std::shared_ptr<mico::Dataframe<pcl::PointXYZRGBNormal>> df = nullptr;

                                            // check data received
                                            try{
                                                df = _data.get<std::shared_ptr<mico::Dataframe<pcl::PointXYZRGBNormal>>>("Dataframe");
                                                image = df->leftImage().clone();
                                                
                                            }catch(std::exception& e){
                                                std::cout << "Failure Darknet dataframe registration. " <<  e.what() << std::endl;
                                                idle_ = true;
                                                return;
                                            }
                                            // auto strt = std::chrono::steady_clock::now();

                                            // vector of detected entities 
                                            std::vector<std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>> entities;

                                            // get image detections
                                            auto detections = detector_.detect(image);
                                            // detection -> label, confidence, left, top, right, bottom

                                            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr featureCloud = df->featureCloud();
                                            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr denseCloud = df->cloud();
                                            std::vector<cv::Point2f> featureProjections = df->featureProjections();

                                            for(auto &detection: detections){
                                               if(detection[1] > confidenceThreshold_){
                                                    std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>> e(new dnn::Entity<pcl::PointXYZRGBNormal>(
                                                         numEntities_, df->id(), detection[0], detection[1], {detection[2],detection[3],detection[4],detection[5]}));  
                                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr entityCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                                                    std::vector<cv::Point2f> entityProjections;

                                                    if(featureProjections.size() > 0 && featureCloud != nullptr){

                                                        if(!useDenseCloud_){ // feature cloud
                                                            for(auto it = featureProjections.begin(); it != featureProjections.end(); it++ ){
                                                                if( it->x > detection[2] && it->x < detection[4] && it->y > detection[3] && it->y < detection[5]){
                                                                    entityProjections.push_back(*it);
                                                                    auto index = it - featureProjections.begin();
                                                                    entityCloud->push_back(featureCloud->points[index]);
                                                                    // mising descriptors
                                                                }
                                                            }
                                                        }
                                                        else{
                                                            // dense cloud
                                                            for (int dy = detection[3]; dy < detection[5]; dy++) {
                                                                for (int dx = detection[2]; dx < detection[4]; dx++) {
                                                                    pcl::PointXYZRGBNormal p = denseCloud->at(dx,dy);
                                                                    if(!boost::math::isnan(p.x) && !boost::math::isnan(p.y) && !boost::math::isnan(p.z)){
                                                                        if(!boost::math::isnan(-p.x) && !boost::math::isnan(-p.y) && !boost::math::isnan(-p.z))
                                                                            entityCloud->push_back(p);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        cv::Rect rec(detection[2], detection[3], detection[4] -detection[2], detection[5]-detection[3]);
                                                        //cv::putText(image, "Confidence" + std::to_string(detection[1]), cv::Point2i(detection[2], detection[3]),1,2,cv::Scalar(0,255,0));
                                                        cv::putText(image, "ObjectId: " + std::to_string(detection[0]), cv::Point2i(detection[2], detection[3]),1,2,cv::Scalar(0,255,0));
                                                        cv::rectangle(image, rec, cv::Scalar(0,255,0));
                                                        
                                                        e->projections(df->id(), entityProjections);

                                                        if(entityCloud->size() > 400){


                                                            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                                                            if(radiusFilter_){
                                                                std::cout << "[BlockDarknet]Starting radius removal" << std::endl;
                                                                mico::radiusFilter<pcl::PointXYZRGBNormal>(entityCloud, cloud_out, radiusSearch_, minNeighbors_);
                                                            }else if(minCutFilter_){
                                                                std::cout << "[BlockDarknet]Starting min cut removal" << std::endl;
                                                                // get point in the center of the bounding box
                                                                int width = detection[4] - detection[2];
                                                                int heigth = detection[5] - detection[3];
                                                                int cx = detection[2] + width / 2;
                                                                int cy = detection[3] + heigth / 2;
                                                                pcl::PointXYZRGBNormal center = denseCloud->at(cx,cy);
                                                                
                                                                // estimate entity radius
                                                                int x1 = detection[2] + width * 0.1;
                                                                int y1 = detection[3] + heigth * 0.1;
                                                                int x2 = detection[2] + width * 0.9;
                                                                int y2 = detection[3] + heigth * 0.9;
                                                                pcl::PointXYZRGBNormal p1 = denseCloud->at(x1,y1);          //    p1----p2
                                                                pcl::PointXYZRGBNormal p2 = denseCloud->at(x2,y1);          //    |      |
                                                                pcl::PointXYZRGBNormal p3 = denseCloud->at(x1,y2);          //    |      |
                                                                pcl::PointXYZRGBNormal p4 = denseCloud->at(x2,y2);          //    p3----p4
                                                                
                                                                std::vector<float> distance;
                                                                distance.push_back(pcl::geometry::distance(p1,p2));
                                                                distance.push_back(pcl::geometry::distance(p1,p3));
                                                                distance.push_back(pcl::geometry::distance(p1,p4));
                                                                distance.push_back(pcl::geometry::distance(p2,p3));
                                                                distance.push_back(pcl::geometry::distance(p2,p4));
                                                                distance.push_back(pcl::geometry::distance(p3,p4));
                                                                sort(distance.begin(), distance.end());
                                                                float radius = (distance[distance.size() / 2 - 1] + distance[distance.size() / 2]) / 2;
                                                                
                                                                // min cut clustering
                                                                mico::minCutSegmentation<pcl::PointXYZRGBNormal>(entityCloud, cloud_out, center, radius, 
                                                                                                                minNeighbors_, weightCutFilter_, sigmaCutFilter_);
                                                                
                                                                float removed = (float)cloud_out->points.size() / (float)entityCloud->points.size();
                                                                std::cout << "[BlockDarknet]Removed " << " input cloud: " << entityCloud->points.size()
                                                                        << " output cloud: " << cloud_out->points.size()
                                                                        << " %  " << removed << " indices" << std::endl;
                                                            }

                                                            e->cloud(df->id(), cloud_out);
                                                            Eigen::Matrix4f dfPose = df->pose();
                                                            e->updateCovisibility(df->id(), dfPose);

                                                            if(cloud_out->size() > 100){
                                                                if(e->computePose(df->id())){
                                                                    entities.push_back(e);
                                                                    std::cout << "[BlockDarknet]Created Entity: " << e->id() << std::endl;
                                                                    if(storeClouds_){
                                                                        std::string fileName = "Entity" + boost::to_string(numEntities_) + ".pcd";
                                                                        pcl::io::savePCDFileASCII(fileName, *cloud_out);
                                                                    }
                                                                    numEntities_++;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            // send entities
                                            if(entities.size() > 0 && getPipe("Entities")->registrations() !=0 )
                                                getPipe("Entities")->flush(entities);
                                            // send image with detections
                                            if(getPipe("Color Image")->registrations() !=0 ){
                                                
                                                getPipe("Color Image")->flush(image);
                                            }
                                            //auto end = std::chrono::steady_clock::now();
                                            //printf("Detector: Elapsed time in milliseconds : %i", std::chrono::duration_cast<std::chrono::milliseconds>(end - strt).count());
                                        }else{
                                            std::cout << "No weights and cfg provided to Darknet\n";
                                        }
                                        #endif
                                        idle_ = true;
                                    }
                                });
    }

    bool BlockDarknet::configure(std::unordered_map<std::string, std::string> _params){        
        #ifdef HAS_DARKNET
        std::string cfgFile;
        std::string weightsFile;
        std::string namesFile;
        for(auto &p: _params){
            if(p.first == "cfg"){
                cfgFile = p.second;
            }else if(p.first == "weights"){
                weightsFile = p.second;
            }else if(p.first == "names"){
                namesFile = p.second;
            }else if(p.first == "confidence_threshold"){
                if(p.second.compare("confidence_threshold") && p.second != ""){
                    std::istringstream istr(_params["confidence_threshold"]);
                    istr >> confidenceThreshold_;
                }
            }else if(p.first == "dense_cloud"){
                if(!p.second.compare("true")){
                    useDenseCloud_ = true;
                }else{
                    useDenseCloud_ = false;
                }
            }else if(p.first == "radius_filter"){
                if(!p.second.compare("true")){ 
                    radiusFilter_ = true;
                }else{
                    radiusFilter_ = false;
                }
            }else if(p.first == "radius_search"){
                if(p.second.compare("radius_search") && p.second != ""){
                    std::istringstream istr(_params["radius_search"]);
                    istr >> radiusSearch_;
                }
            }
            else if(p.first == "minimum_neighbors"){
                if(p.second.compare("minimum_neighbors") && p.second != ""){
                    std::istringstream istr(_params["minimum_neighbors"]);
                    istr >> minNeighbors_;
                }
            }
            else if(p.first == "Min_cut_filter"){
                if(!p.second.compare("true")){
                    minCutFilter_ = true;
                }else{
                    minCutFilter_ = false;
                }
            }else if(p.first == "store_clouds"){
                if(!p.second.compare("true")){
                    storeClouds_ = true;
                }else{
                    storeClouds_ = false;
                }
            }   
        }

        // cfg file provided?
        if(!cfgFile.compare("cfg") || !cfgFile.compare("")){
            std::cout << "[Block Darknet] Cfg not provided \n";                    
            cfgFile = getenv("HOME") + std::string("/.mico/downloads/yolov3-tiny.cfg");
            // cfg file already downloaded?
            if(!std::experimental::filesystem::exists(cfgFile)){
                std::cout << "[Block Darknet] Downloading yolov3-tiny.cfg \n";
                system("wget -P ~/.mico/downloads https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg");
            }
        }   

        // weights file provided?
        if(!weightsFile.compare("weights") || !weightsFile.compare("")){    
            std::cout << "[Block Darknet]Weights not provided \n";                    
            weightsFile = getenv("HOME") + std::string("/.mico/downloads/yolov3-tiny.weights");
            // cfg file already downloaded?
            if(!std::experimental::filesystem::exists(weightsFile)){
                std::cout << "[Block Darknet]Downloading yolov3-tiny.weights \n";
                system("wget -P ~/.mico/downloads https://pjreddie.com/media/files/yolov3-tiny.weights");
            }
        }
        if(!namesFile.compare("names") || !namesFile.compare("")){    
            std::cout << "[Block Darknet] Objects names not provided \n";                    
        }else{
            objects_names_from_file(namesFile);
        }

        std::cout << "[Block Darknet] Cfg file : " << cfgFile << "\n";
        std::cout << "[Block Darknet] Weights file : " << weightsFile << "\n";
        std::cout << "[Block Darknet] Object names file : " << namesFile << "\n";
        std::cout << "[Block Darknet] Confidence threshold : " << confidenceThreshold_ << "\n";
        std::cout << "[Block Darknet] Use dense cloud : " << useDenseCloud_ << "\n";

        hasParameters_ = true;  
        if(detector_.init(cfgFile,weightsFile)){
            return true;
        }
        else{
            std::cout << "Detector: Bad input arguments\n";
            return false;
        }
        #else
        return false;
        #endif
    }
    
    std::vector<std::string> BlockDarknet::parameters(){
        return {"cfg", "weights", "names", "confidence_threshold", "dense_cloud", "radius_filter", "radius_search", "minimum_neighbors","Min_cut_filter","store_clouds"};
    }

    void BlockDarknet::objects_names_from_file(std::string const filename){
        std::ifstream file(filename);
        std::vector<std::string> obj_names_;
        for(std::string line; getline(file, line);) obj_names_.push_back(line);
        std::cout << "[Block Darknet] Objects names loaded \n";
    }

}
