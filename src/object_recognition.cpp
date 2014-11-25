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

#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include <ras_msgs/RAS_Evidence.h>
#include <image_transport/image_transport.h>
using std::cout;
using std::endl;

class object_recognition {
public:
    object_recognition() :
        _it(nh){
        img_path_sub = nh.subscribe("/object_recognition/imgpath", 1, &object_recognition::imgFileCB, this);
        imagedir = "/home/marco/catkin_ws/src/object_recognition/sample_images/";
        img_sub = _it.subscribe("/object_recognition/filtered_image",1, &object_recognition::recognitionCB,this);
        espeak_pub= nh.advertise<std_msgs::String>("/espeak/string",1);
        cv::namedWindow("Image_got_from_detection");
        lastobject= ros::Time::now();
        evidence_pub = nh.advertise<ras_msgs::RAS_Evidence>("/evidence",1);
    }
    void recognitionCB(const sensor_msgs::ImageConstPtr& img_msg){

        //cout<< "got in CB"<< endl;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        //cout<< "loaded pointer"<< endl;

        cv::Mat showimage;
        cv::resize(cv_ptr->image,showimage,cv::Size(400,400),cv::INTER_AREA);
        cv::cvtColor(showimage,showimage,CV_HSV2BGR);
        cv::imshow("Image_got_from_detection",showimage);

        cv::waitKey(1);
        cv::resize(cv_ptr->image,cv_ptr->image,cv::Size(sample_size_x,sample_size_y),cv::INTER_AREA);
        cv::Mat rowImg = matToFloatRow(cv_ptr->image);
        //cout<< rowImg.type()<< endl;

        //PCA
        cv::Mat pcaRowImg;

        pca.project(rowImg,pcaRowImg);

        cv::Mat res;
        kc.find_nearest(pcaRowImg,5,&res);
        std::string result =intToDesc[res.at<float>(0)];

        ros::Time time = ros::Time::now();
        if(!result.compare(("background")) && time.sec-lastobject.sec > 5){
            ras_msgs::RAS_Evidence msg;
            msg.stamp =ros::Time::now();
            msg.object_id = result;
            msg.group_number = 3;
            msg.image_evidence = cv_bridge::CvImage(std_msgs::Header(),"bgr8",showimage).toImageMsg().operator *() ;
            evidence_pub.publish(msg);
            speakresult(result);
            lastobject = time;


        }

    }

    void trainPCA(cv::Mat& rowImg, cv::Mat& result){
        pca = cv::PCA(rowImg,cv::Mat(), CV_PCA_DATA_AS_ROW,0.95);
        pca.project(rowImg,result);

    }

    void imgFileCB(const std_msgs::String& pathToImg) {
        cout << "Classifying " << pathToImg.data << endl;
        cv::Mat inputImg = cv::imread(pathToImg.data);
        cv::Mat showimage;
        cv::cvtColor(inputImg,showimage,CV_HSV2BGR);
        cv::resize(showimage,showimage,cv::Size(400,400));

        cv::imshow("Image_got_from_detection",showimage);

        cv::waitKey(1);

        cv::Mat rowImg = matToFloatRow(inputImg);
        cv::Mat pcaRowImg;
        pca.project(rowImg,pcaRowImg);
        cv::Mat res;
        cout<< "Before PCA attributes " << rowImg.cols << "After PCA attributes " << pcaRowImg.cols << endl;
        kc.find_nearest(pcaRowImg, 3, &res);
        std::string result =intToDesc[res.at<float>(0)];
        ros::Time time = ros::Time::now();
        if(0!=result.compare(("background")) && time.sec-lastobject.sec >5){
            ras_msgs::RAS_Evidence msg;
            msg.stamp =ros::Time::now();
            msg.object_id = result;
            msg.group_number = 3;
            msg.image_evidence = cv_bridge::CvImage(std_msgs::Header(),"bgr8",showimage).toImageMsg().operator *() ;
            evidence_pub.publish(msg);
            speakresult(result);
            lastobject = time;

        }

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
        cv::Mat pcatrainData;
        trainPCA(trainData,pcatrainData);
        std::cout << "Try to train"<< std::endl;
        kc.train(pcatrainData, responses);
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
    void speakresult(std::string detectedobject){
        std::stringstream ss;

        ss<<"I see a " << detectedobject;
        std_msgs::String msg;
        msg.data=ss.str();
        espeak_pub.publish(msg);
    }

private:
    cv::PCA pca;
    ros::Publisher espeak_pub , evidence_pub;
    ros::NodeHandle nh;
    ros::Subscriber img_path_sub;
    static const int sample_size_x = 100;
    static const int sample_size_y = 100;
    static const int attributes = 2;
    std::map<int, std::string> intToDesc;
    std::string imagedir;
    cv::KNearest kc;
    image_transport::ImageTransport _it;
    image_transport::Subscriber img_sub;
    ros::Time lastobject;
};


int main(int argc, char** argv){
    ros::init(argc, argv, "object_recognition");
    object_recognition object_rec;
    object_rec.train_knn();
    ros::Rate rate(1);
    while(ros::ok()){
        ros::spinOnce();
    }
}
