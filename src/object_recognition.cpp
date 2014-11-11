#include <ros/ros.h>
#include <cstdio>
#include <iostream>
#include <map>
#include <ostream>
#include <std_msgs/String.h>
#include <dirent.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using std::cout;
using std::endl;

class object_recognition {
public:
    object_recognition() {
        img_path_sub = nh.subscribe("/object_recognition/imgpath", 1, &object_recognition::imgFileCB, this);
        imagedir = "/home/ras/catkin_ws/src/object_recognition/sample_images/";
    }

    void imgFileCB(const std_msgs::String& pathToImg) {
        cout << "Classifying " << pathToImg.data << endl;
        cv::Mat inputImg = cv::imread(pathToImg.data);
        cv::Mat rowImg = matToFloatRow(inputImg);
        cv::Mat res;
        kc.find_nearest(rowImg, 5, &res);
        cout << intToDesc[res.at<float>(0)] << endl;
    }

    void train_knn(){
        std::vector<std::pair<std::string, std::vector<std::string> > > objects = readTestImagePaths(imagedir);
        cv::Mat trainData;
        cv::Mat responses;
        for(int i = 0; i < objects.size(); i++) {
            cout << objects[i].first << " = " << i << endl;
            intToDesc[i] = objects[i].first;
            std::vector<std::string>& vec = objects[i].second;
            for(int j = 0; j < vec.size(); j++) {
                responses.push_back(i);
                cv::Mat inputImg = cv::imread(imagedir + objects[i].first + "/" + vec[j]);
                cv::Mat rowImg = matToFloatRow(inputImg);
                trainData.push_back(rowImg);
            }
        }

        std::cout << "Try to train"<< std::endl;
        kc.train(trainData, responses);
        std::cout<< "Training succeded"<< std::endl;
    }


    std::vector<std::pair<std::string, std::vector<std::string> > >
    readTestImagePaths(std::string directory) {
        std::vector<std::pair<std::string, std::vector<std::string> > > objects;
        DIR* dirPtr;
        dirent* entry;

        if((dirPtr = opendir(directory.c_str())) == NULL) {
            cout << "Could not open directory for training";
            return objects;
        }

        entry = readdir(dirPtr);
        while(entry != NULL) {
            if(entry->d_type == DT_DIR && entry->d_name[0] != '.') objects.push_back(make_pair(entry->d_name, std::vector<std::string>()));
            entry = readdir(dirPtr);
        }

        closedir(dirPtr);
        for(int i = 0; i < objects.size(); i++) {
            dirPtr = opendir((directory + objects[i].first).c_str());
            entry = readdir(dirPtr);
            while(entry != NULL) {
                if(entry->d_type != DT_DIR) objects[i].second.push_back(entry->d_name);
                entry = readdir(dirPtr);
            }
        }

        return objects;
    }

    cv::Mat matToFloatRow(const cv::Mat& input) {
        cv::Mat res(1, input.rows*input.cols*attributes, CV_32FC1);
        int rows = input.rows;
        int cols = input.cols;
        for(int x=0; x < rows; x++){
            for (int y=0; y<cols; y++){
                res.at<float>(0,((x*cols + y)*attributes))      = float(input.at<cv::Vec3b>(x,y)[0]);
                res.at<float>(0,((x*cols + y)*attributes + 1)) = float(input.at<cv::Vec3b>(x,y)[1]);
            }
        }
        return res;
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber img_path_sub;
    static const int sample_size_x = 2;
    static const int sample_size_y = 2;
    static const int attributes = 2;
    std::map<int, std::string> intToDesc;
    std::string imagedir;
    cv::KNearest kc;
};


int main(int argc, char** argv){
    ros::init(argc, argv, "object_recognition");
    object_recognition object_rec;
    object_rec.train_knn();
    ros::spin();
}
