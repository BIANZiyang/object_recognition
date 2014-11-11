#include <ros/ros.h>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

enum Objects { REDCUBE,BLUECUBE,GREENCUBE,YELLOWCUBE,YELLOWBALL,REDBALL,GREENCYLINDER,BLUETRIANGLE,PURPLECROSS,PATRIC };
class object_recognition{
public:
    object_recognition(){
        //pcl_sub = nh.subscribe("/object_recognition/image", 1, &object_recognition::recognitionCb, this);
        sample_image_count=10;
    }
    void recognitionCB(){


        Object_output(REDCUBE);
    }

    void train_Bayes_Classifier(){
        std::cout << "Try to allocate disk space"<< std::endl;
        cv::Mat trainData(sample_image_count,(sample_size_x*sample_size_y*2),CV_32FC1);
        cv::Mat responses(sample_image_count,1,CV_32SC1);
        std::cout <<" allocated disk space"<< std::endl;
        for(int i=0;i<sample_image_count;i++){
            std::stringstream ss ;
            ss << "sample_images/"<< i << "sample.ppm" ;

            cv::Mat image= cv::imread(ss.str());
            //Class 0 will be no object, Class 1 will be object 1, class 2 object 2 ...
            if(i<5){
                responses.at<int>(i)=0;
            }
            else{
                responses.at<int>(i)=1;
            }
            for(int y=0;y<sample_size_y;y++){
                for (int x=0;x<sample_size_x;x++){
                    trainData.at<float>(i,(y*sample_size_x*2+x*2))=image.at<cv::Vec3b>(x,y)[0];
                    trainData.at<float>(i,(y*sample_size_x*2+x*2+1))=image.at<cv::Vec3b>(x,y)[1];
                }
            }
        }
        std::cout << "loaded Data"<< std::endl;
        cv::NormalBayesClassifier bc;
        std::cout << "Try to train"<< std::endl;

        bc.train(trainData,responses);
        std::cout<< "Training succeded"<< std::endl;
    }

private:
    void reshape_image(cv::Mat& src, cv::Mat& dst ){
        cv::Size dsize = cv::Size(sample_size_x,sample_size_y);
        cv::resize(src,dst,dsize,cv::INTER_AREA);
        cv::namedWindow("Display Window");
        imshow("Display Window",dst);
        cv::waitKey();
    }

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

    int sample_image_count;
    static const int sample_size_x = 10;
    static const int sample_size_y = 10;
    cv::NormalBayesClassifier bc;
};





int main(int argc, char** argv){
    ros::init(argc, argv, "object_recognition");
    object_recognition object_rec;
    object_rec.train_Bayes_Classifier();
    //ros::spin();
    //object_rec.recognitionCB();
}
