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


namespace dnn {

    template<typename PointType_>
    inline Entity<PointType_>::Entity(int _id, std::shared_ptr<mico::Dataframe<PointType_>> _df, int _label, float _confidence, std::vector<float> _boundingbox){
        id_ = _id;
        label_ = _label;
        confidence_[_df->id()] = _confidence;
        boundingbox_[_df->id()] = _boundingbox;
        dfs_.push_back(_df->id());
        dfMap_[_df->id()] = _df;
    }

    template<typename PointType_>
    inline Entity<PointType_>::Entity(int _id, int _label, float _confidence, std::vector<float> _boundingbox){
        id_ = _id;
        label_ = _label;
        confidence_[0] = _confidence;
        boundingbox_[0] = _boundingbox;
        dfs_.push_back(0);
    }

    template<typename PointType_>
    inline bool Entity<PointType_>::computePose(int _dataframeId){   // 666 _dataframeId to compute pose with several clouds        
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        std::vector<float> bc;
        // Compute principal directions
        if(!mico::computePCA(*denseClouds_[_dataframeId], pose, bc))
            return false;

        poses_[_dataframeId] = pose;   // to global pose 666 check this
        // poses_[_dataframeId] = pose * covisibility_[_dataframeId];   // to global pose 666 check this
        positions_[_dataframeId] = pose.block(0,3,3,1);
        Eigen::Quaternionf q(pose.block<3,3>(0,0));
        orientations_[_dataframeId] = q;
        boundingcube_[_dataframeId] = bc;

        float width = bc[0] - bc[1]; 
        float heigth = bc[2] - bc[3]; 
        float deep = bc[4] - bc[5];
        auto globalCubePose = covisibility_[_dataframeId] * pose;
        cube_ = std::make_shared<Cube>(1, globalCubePose, width, heigth, deep);
        return true;
    }

    template<typename PointType_>
    inline void Entity<PointType_>::updateCovisibility(int _dataframeId, Eigen::Matrix4f &_pose){
        // std::lock_guard<std::mutex> lock(dataLock_);
        covisibility_[_dataframeId] = _pose;

        // check for new dataframe
        if(std::find(dfs_.begin(), dfs_.end(), _dataframeId) == dfs_.end())
            dfs_.push_back(_dataframeId);
    }
    
    template<typename PointType_>
    inline int Entity<PointType_>::id() const{
        // std::lock_guard<std::mutex> lock(dataLock_);
        return id_;
    }

    template<typename PointType_>
    inline void Entity<PointType_>::pose(int _dataframeId, Eigen::Matrix4f &_pose){
            // std::lock_guard<std::mutex> lock(dataLock_);
            poses_[_dataframeId]   = _pose;
            positions_[_dataframeId]   = _pose.block<3,1>(0,3);
            orientations_[_dataframeId]   = Eigen::Quaternionf(_pose.block<3,3>(0,0));
    }

    template<typename PointType_>
    inline Eigen::Matrix4f Entity<PointType_>::pose(int _dataframeId){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return poses_[_dataframeId];
    }

    template<typename PointType_>
    inline void Entity<PointType_>::dfpose(int _dataframeId, Eigen::Matrix4f &_pose){
        // std::lock_guard<std::mutex> lock(dataLock_);
        covisibility_[_dataframeId]   = _pose;
    }

    template<typename PointType_>
    inline Eigen::Matrix4f Entity<PointType_>::dfpose(int _dataframeId){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return covisibility_[_dataframeId];
    }

    template<typename PointType_>
    inline void Entity<PointType_>::projections(int _dataframeId, std::vector<cv::Point2f> _projections){
        // std::lock_guard<std::mutex> lock(dataLock_);
        projections_[_dataframeId] = _projections;
        // check for new dataframe
        if(std::find(dfs_.begin(), dfs_.end(), _dataframeId) == dfs_.end())
            dfs_.push_back(_dataframeId);
    }

    template<typename PointType_>
    inline std::vector<cv::Point2f> Entity<PointType_>::projections(int _dataframeId){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return projections_[_dataframeId];
    }

    template<typename PointType_>
    inline void Entity<PointType_>::descriptors(int _dataframeId, cv::Mat _descriptors){
        // std::lock_guard<std::mutex> lock(dataLock_);
        descriptors_[_dataframeId] = _descriptors.clone();
        // check for new dataframe
        if(std::find(dfs_.begin(), dfs_.end(), _dataframeId) == dfs_.end())
            dfs_.push_back(_dataframeId);
    }

    template<typename PointType_>
    inline cv::Mat Entity<PointType_>::descriptors(int _dataframeId){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return descriptors_[_dataframeId];
    }

    template<typename PointType_>
    inline void Entity<PointType_>::cloud(int _dataframeId, typename pcl::PointCloud<PointType_>::Ptr &_cloud){
        // std::lock_guard<std::mutex> lock(dataLock_);
        denseClouds_[_dataframeId] = _cloud;
        // check for new dataframe
        if(std::find(dfs_.begin(), dfs_.end(), _dataframeId) == dfs_.end())
            dfs_.push_back(_dataframeId);
    }

    template<typename PointType_>
    inline typename pcl::PointCloud<PointType_>::Ptr  Entity<PointType_>::cloud(int _dataframeId){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return denseClouds_[_dataframeId];
    }

    template<typename PointType_>
    inline void Entity<PointType_>::featureCloud(int _dataframeId, typename pcl::PointCloud<PointType_>::Ptr &_cloud){
        // std::lock_guard<std::mutex> lock(dataLock_);
        featureClouds_[_dataframeId] = _cloud;
        // check for new dataframe
        if(std::find(dfs_.begin(), dfs_.end(), _dataframeId) == dfs_.end())
            dfs_.push_back(_dataframeId);
    }

    template<typename PointType_>
    inline typename pcl::PointCloud<PointType_>::Ptr  Entity<PointType_>::featureCloud(int _dataframeId){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return featureClouds_[_dataframeId];
    }

    template<typename PointType_>
    inline void Entity<PointType_>::boundingbox(int _dataframeId, std::vector<float> _bb){
        // std::lock_guard<std::mutex> lock(dataLock_);
        boundingbox_[_dataframeId] = _bb;
        // check for new dataframe
        if(std::find(dfs_.begin(), dfs_.end(), _dataframeId) == dfs_.end())
            dfs_.push_back(_dataframeId);
    }; 

    template<typename PointType_>
    inline std::vector<float> Entity<PointType_>::boundingbox(int _dataframeId){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return boundingbox_[_dataframeId];
    };

    template<typename PointType_>    
    inline void Entity<PointType_>::boundingCube(int _dataframeId, std::vector<float> _bc){
        // std::lock_guard<std::mutex> lock(dataLock_);
        boundingcube_[_dataframeId] = _bc;
        if(std::find(dfs_.begin(), dfs_.end(), _dataframeId) == dfs_.end())
            dfs_.push_back(_dataframeId);
    };

    template<typename PointType_>    
    inline std::vector<float> Entity<PointType_>::boundingCube(int _dataframeId){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return boundingcube_[_dataframeId];
    };

    template<typename PointType_>    
    inline std::vector<int> Entity<PointType_>::dfs(){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return dfs_;
    };

    template<typename PointType_>    
    inline std::shared_ptr<Cube> Entity<PointType_>::cube(){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return cube_;
    };

    template<typename PointType_>    
    inline float Entity<PointType_>::percentageOverlapped(std::shared_ptr<dnn::Entity<PointType_>> _e){
        // std::lock_guard<std::mutex> lock(dataLock_);
        std::vector<Eigen::Vector3f> inter;
        auto queryCube = _e->cube();
        // compute intersection of two cubes
        queryCube->clipConvexPolyhedron(cube_, inter);
        // calculate volume overlapped
        float percentage = queryCube->computeVolumeFromPoints(inter) / queryCube->getVolume();
        float selfPercentage = cube_->computeVolumeFromPoints(inter) / cube_->getVolume();
        if(selfPercentage < percentage)
            return selfPercentage;
        return percentage;
    };

    template<typename PointType_>    
    inline void Entity<PointType_>::update(std::shared_ptr<dnn::Entity<PointType_>> _e){
        // std::lock_guard<std::mutex> lock(dataLock_);
        for(auto &df: _e->dfs()){
            confidence_[df] = _e->confidence(df);
            boundingCube(df, boundingCube(df));
            dfs_.push_back(df);
        }
    }

    template<typename PointType_>    
    inline void Entity<PointType_>::confidence(int _df, float _confidence){
        // std::lock_guard<std::mutex> lock(dataLock_);
        if ( confidence_.find(_df) == confidence_.end() ) {
            // not found 
            confidence_[_df] = _confidence;
        } else {
            // found
            std::cerr << "[Entity] Trying to change the confidences of an already created object" << std::endl;
        }
    }

    template<typename PointType_>    
    inline float Entity<PointType_>::confidence(int _df){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return confidence_[_df];
    }

    template<typename PointType_>    
    inline std::string Entity<PointType_>::name(){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return name_;
    }

    template<typename PointType_>    
    inline void Entity<PointType_>::name(std::string _name, cv::Scalar _color){
        // std::lock_guard<std::mutex> lock(dataLock_);
        name_ = _name;
        color_ = _color;
    }

    template<typename PointType_>    
    inline cv::Scalar Entity<PointType_>::color(){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return color_;
    }

    template<typename PointType_>    
    inline int Entity<PointType_>::label(){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return label_;
    }

    template<typename PointType_>
    inline std::map<int , std::map<int, std::vector<cv::DMatch>>>& Entity<PointType_>::crossReferencedInliers(){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return multimatchesInliersDfs_;
    }

    template<typename PointType_>
    inline  std::map<int, std::shared_ptr<mico::Word<PointType_>>> Entity<PointType_>::words(){
        // std::lock_guard<std::mutex> lock(dataLock_);
        return wordsReference_;
    }

    template<typename PointType_>
    inline void Entity<PointType_>::addWord(const std::shared_ptr<mico::Word<PointType_>> &_word){
        // std::lock_guard<std::mutex> lock(dataLock_);
        wordsReference_[_word->id] = _word;
    }

    template<typename PointType_>
    inline void Entity<PointType_>::createWords(){
        // try to create words with every dataframe related with other trought the multi matches inliers
        for(auto &firstDf: multimatchesInliersDfs_){
            for(auto &secondDf: firstDf.second){
                // if match not computed
                if(!( createdWordsBetweenDfs_.find(std::make_pair(firstDf.first, secondDf.first)) == createdWordsBetweenDfs_.end() )){
                    if(createdWordsBetweenDfs_[std::make_pair(firstDf.first, secondDf.first)] == false){
                        wordCreation(firstDf.first, secondDf.first);
                        createdWordsBetweenDfs_[std::make_pair(firstDf.first, secondDf.first)] = true;
                        createdWordsBetweenDfs_[std::make_pair(secondDf.first, firstDf.first)] = true;
                    }
                }
            }
        }
    }

    template<typename PointType_>    
    inline void Entity<PointType_>::wordCreation(int _queryDfId, int _trainDfId){         // newer-older  self-previous

        // check if its posible
        if ( !(multimatchesInliersDfs_.find(_queryDfId) == multimatchesInliersDfs_.end()) ) {
            std::cerr << "[Entity] Trying to create words in entity " << id_ << " and not seen by dataframe " << _queryDfId << std::endl;
            return;
        }
        // get matches of the dataframe  
        std::map<int, std::vector<cv::DMatch>> dfMatches = multimatchesInliersDfs_[_queryDfId];

        if ( !(dfMatches.find(_trainDfId) == dfMatches.end()) ) {
            std::cerr << "[Entity] Trying to create words in entity " << id_ << " and not seen by dataframe " << _trainDfId << std::endl;
            return;
        }

        // try to create new words or update pre-existing words with the rest of dataframes in the entity
        std::vector<cv::DMatch> cvInliers = dfMatches[_trainDfId];

        // transform feature cloud of last dataframe feature cloud seen
        typename pcl::PointCloud<PointType_>::Ptr transformedFeatureCloud(new pcl::PointCloud<PointType_>());
        pcl::transformPointCloud(*featureCloud(_trainDfId), *transformedFeatureCloud, dfpose(_trainDfId));

        for (unsigned inlierIdx = 0; inlierIdx < cvInliers.size(); inlierIdx++){
            std::shared_ptr<mico::Word<PointType_>> prevWord = nullptr;
            int inlierIdxInQuery = cvInliers[inlierIdx].queryIdx;       
            int inlierIdxInTrain = cvInliers[inlierIdx].trainIdx;        

            // Check if exists a word with the id of the descriptor inlier in the entity 
            for (auto &w : wordsReference_){ 
                if (w.second->idxInDf[_trainDfId] == inlierIdxInTrain) {
                    prevWord = w.second;
                    break;
                }
            }
            if (prevWord) {
                if (prevWord->dfMap.find(_queryDfId) == prevWord->dfMap.end()) {
                    std::vector<float> projection = {   projections(_queryDfId)[inlierIdxInQuery].x,
                                                        projections(_queryDfId)[inlierIdxInQuery].y};
                    prevWord->addObservation(dfMap_[_queryDfId], inlierIdxInQuery, projection);    /// TODO
                    addWord(prevWord);
                    
                    // 666 CHECK IF IT IS NECESARY
                    for (auto &df : prevWord->dfMap) {
                        dfMap_[_queryDfId]->appendCovisibility(df.second); // TODO -------------------------------------------------------------------------------------
                        // Add current dataframe id to others dataframe covisibility
                        dfMap_[_trainDfId]->appendCovisibility(dfMap_[_queryDfId]); // TODO -------------------------------------------------------------------------------------
                    }
                }
            }
            else {
                // Create word
                int wordId = 0;
                if(wordsReference_.size()>0)
                    wordId = wordsReference_.size() + 1; 

                auto pclPoint = (*transformedFeatureCloud)[inlierIdxInTrain];
                std::vector<float> point = {pclPoint.x, pclPoint.y, pclPoint.z};
                auto descriptor = descriptors(_trainDfId).row(inlierIdxInTrain);
                auto newWord = std::shared_ptr<mico::Word<PointType_>>(new mico::Word<PointType_>(wordId, point, descriptor));

                // Add word to new dataframe (new dataframe is representative of the new dataframe)
                std::vector<float> dataframeProjections = { projections(_queryDfId)[inlierIdxInQuery].x, 
                                                            projections(_queryDfId)[inlierIdxInQuery].y};
                newWord->addObservation(dfMap_[_queryDfId], inlierIdxInQuery, dataframeProjections);       // TODO -------------------------------------------------------------------------------------
                // appendCovisibility(dfMap[_trainDfId]);  // TODO -------------------------------------------------------------------------------------
                addWord(newWord);

                // Add word to last dataframe
                std::vector<float> projection = {   projections(_trainDfId)[inlierIdxInTrain].x, 
                                                    projections(_trainDfId)[inlierIdxInTrain].y};
                // newWord->addObservation(dfMap[_trainDfId], inlierIdxInTrain, projection); // TODO -------------------------------------------------------------------------------------
                
                // prevDf->appendCovisibility(dfMap[_trainDfId]); 
                dfMap_[_trainDfId]->addWord(newWord);
            }
        }





    }

    
}