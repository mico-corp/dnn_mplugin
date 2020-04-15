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

#ifndef MICO_FLOW_DNN_BLOCKS_MISC_IMPL_ENTITYVISUALIZER_H_
#define MICO_FLOW_DNN_BLOCKS_MISC_IMPL_ENTITYVISUALIZER_H_

#include <QDialog>
#include <QTreeWidget>
#include <QTabWidget>

#include <pcl/point_types.h>

#include <mico/dnn/map3d/Entity.h>
#include <mico/slam/Word.h>


namespace dnn{
    class EntityVisualizer : public QDialog {
    public:
        explicit EntityVisualizer(Entity<pcl::PointXYZRGBNormal>::Ptr , QWidget *parent = 0);

    private:
        QTabWidget *tabWidget_;

    };
}



#endif // EntityVisualizer_H_