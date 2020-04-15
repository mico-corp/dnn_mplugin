//---------------------------------------------------------------------------------------------------------------------
//  MICO
//---------------------------------------------------------------------------------------------------------------------
//  Copyright 2020 Ricardo Lopez Lopez (a.k.a. sanso92) ricloplop@gmail.com
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

#include <mico/dnn/flow/impl/EntityVisualizer.h>
#include <QtWidgets>


namespace dnn{

    class ImageTab : public QWidget{
        public:
            ImageTab(cv::Mat _image, QWidget *parent = nullptr){
                cv::cvtColor(_image,_image,CV_BGR2RGB); //Qt reads in RGB whereas CV in BGR
                QImage imdisplay((uchar*)_image.data, _image.cols, _image.rows, _image.step, QImage::Format_RGB888); 
                imageDisplay_ = new QLabel;
                imageDisplay_->setPixmap(QPixmap::fromImage(imdisplay));
                QHBoxLayout *layout = new QHBoxLayout();
                layout->addWidget(imageDisplay_);
                setLayout(layout);
            }
        private:
            QLabel *imageDisplay_;
    };

    EntityVisualizer::EntityVisualizer(Entity<pcl::PointXYZRGBNormal>::Ptr _e, QWidget *parent){
        QVBoxLayout *mainLayout = new QVBoxLayout;
        setLayout(mainLayout);
        tabWidget_ = new QTabWidget;
        mainLayout->addWidget(tabWidget_);


        if(_e->dfs().size()>1){
            auto dfs = _e->dfMap();

            for(auto firstDf = dfs.begin(); next(firstDf,1) != dfs.end() ; firstDf++){
                for(auto secondDf = next(firstDf,1); secondDf != dfs.end(); secondDf++){
                    auto queryId = (*firstDf).second->id();
                    auto trainId = (*secondDf).second->id();

                    cv::Mat queryImage = dfs[queryId]->leftImage();
                    cv::Mat trainImage = dfs[trainId]->leftImage();
                    
                    for(auto &[wid, word]:_e->words()){
                        if(word->isInFrame(queryId) && word->isInFrame(trainId)){
                            cv::circle(queryImage, word->cvProjectionf(queryId), 3, cv::Scalar(0,255,0));
                            cv::putText(queryImage, std::to_string(word->id) , word->cvProjectionf(queryId), 2, 0.5, cv::Scalar(0,255,0), 0.5);

                            cv::circle(trainImage, word->cvProjectionf(trainId), 3, cv::Scalar(0,255,0));
                            cv::putText(trainImage, std::to_string(word->id), word->cvProjectionf(trainId), 2, 0.5, cv::Scalar(0,255,0), 0.5);
                        }
                    }

                    cv::Mat finalImage;
                    cv::hconcat(queryImage, trainImage, finalImage);

                    std::stringstream label; label << queryId << "--" << trainId;
                    tabWidget_->addTab(new ImageTab(finalImage), label.str().c_str());
                }
            }
        }
    }
}