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

#ifndef MICO_DNN_MAP3D_ENTITY_H_
#define MICO_DNN_MAP3D_ENTITY_H_

#include <string>
#include <cmath>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include <mico/slam/utils3d.h>
#include <mico/dnn/utils/Cube.h>

namespace dnn {

template <typename PointType_>
struct Entity
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    typedef std::shared_ptr<Entity<PointType_>> Ptr;

    Entity(int _id, int _dataframeId, int _label, float _confidence, std::vector<float> _boundingbox);
    Entity(int _id, int _label, float _confidence, std::vector<float> _boundingbox);

    bool computePose(int _dataframeId); // 666 change name

    int id() const;

    void pose(int _dataframeId, Eigen::Matrix4f &_pose);
    Eigen::Matrix4f pose(int _dataframeId);

    void dfpose(int _dataframeId, Eigen::Matrix4f &_pose);
    Eigen::Matrix4f dfpose(int _dataframeId);

    void cloud(int _dataframeId, typename pcl::PointCloud<PointType_>::Ptr &_cloud);
    typename pcl::PointCloud<PointType_>::Ptr cloud(int _dataframeId);

    void boundingbox(int _dataframeId, std::vector<float> _bb);
    std::vector<float> boundingbox(int _dataframeId);

    void boundingCube(int _dataframeId, std::vector<float> _bc);
    std::vector<float> boundingCube(int _dataframeId);

    void projections(int _dataframeId, std::vector<cv::Point2f> _projections);
    std::vector<cv::Point2f> projections(int _dataframeId);

    void descriptors(int _dataframeId, cv::Mat _descriptors);
    cv::Mat descriptors(int _dataframeId);

    std::vector<int> dfs();

    void updateCovisibility(int _dataframeId, Eigen::Matrix4f &_pose);

    std::shared_ptr<Cube> cube();
    
    float percentageOverlapped(std::shared_ptr<dnn::Entity<PointType_>> _e);

    void confidence(int _df, float _confidence);

    float confidence(int _df);

    // update the entity with anoter entity
    void update(std::shared_ptr<dnn::Entity<PointType_>> _e);
    
    cv::Scalar color();

    std::string name();
    
    void name(std::string _name, cv::Scalar _color);

    int label();

private:
    Entity(){};

    size_t id_;
    std::vector<int> dfs_;

    /// pose from dataframe view
    std::map<int, Eigen::Matrix4f> poses_;
    std::map<int, Eigen::Vector3f> positions_;
    std::map<int, Eigen::Quaternionf> orientations_;

    /// 3D
    std::map<int, typename pcl::PointCloud<PointType_>::Ptr> clouds_;

    /// 2D
    std::map<int, std::vector<cv::Point2f>> projections_;
    std::map<int, cv::Mat> descriptors_;

    /// detection
    int label_;
    std::string name_;
    cv::Scalar color_;
    std::map<int, float> confidence_;
    std::map<int, std::vector<float>> boundingbox_;  // left top right bottom
    std::map<int, std::vector<float>> boundingcube_; // xmax xmin ymax ymin zmax zmin

    // cube
    std::shared_ptr<Cube> cube_ = nullptr;

    /// visibility
    std::map<int, Eigen::Matrix4f> covisibility_; // dataframe id and pose
};

template <typename PointType_>
std::ostream &operator<<(std::ostream &os, const Entity<PointType_> &e)
{
    auto dataframes = e.dfs();
    for (auto &df : dataframes)
    {
        os << e->pose(df)(0,3) << ", "<< e->pose(df)(1,3) << ", "<< e->pose(df)(2,3);
    }
    return os;
}
} // namespace dnn
#include <mico/dnn/map3d/Entity.inl>

#endif