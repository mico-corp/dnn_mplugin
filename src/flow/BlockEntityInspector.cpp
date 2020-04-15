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

#include <mico/dnn/flow/BlockEntityInspector.h>

#include <QDialog>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>

#include <mico/dnn/map3d/Entity.h>
#include <mico/dnn/flow/impl/EntityVisualizer.h>

namespace dnn{
    BlockEntityInspector::BlockEntityInspector(){
        createPolicy({{"Entities", "v_entity"}});
        

        registerCallback({ "Entities" }, 
                            [&](flow::DataFlow  _data){
                                auto entities = _data.get<std::vector<std::shared_ptr<dnn::Entity<pcl::PointXYZRGBNormal>>>>("Entities"); 
                                for(auto &e: entities){
                                    entities_[e->id()] = e;
                                    QTreeWidgetItem *eTreeItem = new QTreeWidgetItem(eList_);
                                    eTreeItem->setText(0, std::to_string(e->id()).c_str());
                                }
                            }
                        );
    }
    
    BlockEntityInspector::~BlockEntityInspector(){
    
    }

    QWidget * BlockEntityInspector::customWidget() {
        eList_ = new QTreeWidget();

        
        QWidget::connect(eList_, &QTreeWidget::itemClicked, [this](QTreeWidgetItem* _item, int _id){
            auto id = _item->text(0).toInt();
            EntityVisualizer ev(this->entities_[id]);
            ev.show();
            ev.exec();
        });

        return eList_;
    }

}

