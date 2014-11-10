#include <ros/ros.h>

#include <iostream>
#include <ostream>

enum Objects { REDCUBE,BLUECUBE,GREENCUBE,YELLOWCUBE,YELLOWBALL,REDBALL,GREENCYLINDER,BLUETRIANGLE,PURPLECROSS,PATRIC };
class object_recognition{
public:
    object_recognition(){
        //pcl_sub = nh.subscribe("/object_recognition/image", 1, &object_recognition::recognitionCb, this);
    }
    void recognitionCB(){




        Object_output(REDCUBE);
    }

private:
    void Object_output(Objects const detected){
        std::string output;
        switch(detected){
            case REDCUBE: output =   "Red Cube" ; break;
            case BLUECUBE: output =   "Blue Cube" ; break;
            case GREENCUBE: output =   "Green Cube" ; break;
            case YELLOWCUBE: output =   "Yellow Cube" ; break;
            case YELLOWBALL: output =   "Yellow Ball" ; break;
            case REDBALL: output =   "Red Ball" ; break;
            case GREENCYLINDER: output =   "Green Cylinder" ; break;
            case BLUETRIANGLE: output =   "Blue Triangle" ; break;
            case PURPLECROSS: output =   "Purple Cross" ; break;
            case PATRIC: output =   "Patric" ; break;
            default: output =  "An Object";
        }
        std::cout << output;
    }
};





int main(int argc, char** argv){
    ros::init(argc, argv, "object_detection");
    object_recognition object_rec;
    ros::spin();

}
