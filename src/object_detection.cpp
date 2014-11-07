// Copyed from Hand_tracker
#include <cstdio>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>

#include <opencv2/highgui/highgui.hpp>



#include <pcl/filters/statistical_outlier_removal.h>
typedef pcl::PCLPointCloud2 Cloud2;
typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> Cloud;
typedef pcl::PointXYZHSV PointHSV;
typedef pcl::PointCloud<PointHSV> CloudHSV;

class object_detection{
public:
    object_detection() {
        pcl_sub = nh.subscribe("/camera/depth_registered/points", 1, &object_detection::imageCb, this);
        vis_pub = nh.advertise<sensor_msgs::PointCloud2>("/object_recognition/deteciton", 1);
        load_from_launchfile();
        cv::namedWindow("HSVTrackbars",1);
        cv::createTrackbar("Hmin","HSVTrackbars",&hmin,179);
        cv::createTrackbar("Hmax","HSVTrackbars",&hmax,179);
        cv::createTrackbar("Smin","HSVTrackbars",&smin,255);
        cv::createTrackbar("Smax","HSVTrackbars",&smax,255);
        cv::createTrackbar("Vmin","HSVTrackbars",&vmin,255);
        cv::createTrackbar("Vmax","HSVTrackbars",&vmax,255);
    }
    void load_from_launchfile(){
        voxelsize=0.005;
        hmin=0;
        hmax=120;
        smin=0;
        smax=255;
        vmin=0;
        vmax=255;
        double temp_leafsize;
        double depth = 0.3, height = 0.3, width = 0.3, heightOffset = 0.3, depthOffset = 0.1;
        if(nh.hasParam("object_detection/leafsize")) {nh.getParam("object_detection/leafsize", temp_leafsize); voxelsize=temp_leafsize;}
        if(nh.hasParam("object_detection/depth")) {nh.getParam("object_detection/depth", depth);}
        if(nh.hasParam("object_detection/height")) {nh.getParam("object_detection/height", height);}
        if(nh.hasParam("object_detection/width")) {nh.getParam("object_detection/width", width);}
        if(nh.hasParam("object_detection/heightOffset")) {nh.getParam("object_detection/heightOffset", heightOffset);}
        if(nh.hasParam("object_detection/depthOffset")) {nh.getParam("object_detection/depthOffset", depthOffset);}

        if(nh.hasParam("object_detection/hmin")) {nh.getParam("object_detection/hmin", hmin);}
        if(nh.hasParam("object_detection/hmax")) {nh.getParam("object_detection/hmax", hmax);}
        if(nh.hasParam("object_detection/smin")) {nh.getParam("object_detection/smin", smin);}
        if(nh.hasParam("object_detection/smax")) {nh.getParam("object_detection/smax", smax);}
        if(nh.hasParam("object_detection/vmin")) {nh.getParam("object_detection/vmin", vmin);}
        if(nh.hasParam("object_detection/vmax")) {nh.getParam("object_detection/vmax", vmax);}


        //Order of params: width, height, depth
        minVal[0] = -width/2;
        minVal[1] = -height/2 - heightOffset;
        minVal[2] = -depth/2 + depthOffset;
        maxVal[0] = width/2;
        maxVal[1] = height/2 - heightOffset;
        maxVal[2] = depth/2 + depthOffset;
        lower[0] = hmin;
        lower[1] = smin;
        lower[2] = vmin;
        upper[0] = hmax;
        upper[1] = smax;
        upper[2] = vmax;


    }

    void imageCb(const sensor_msgs::PointCloud2ConstPtr& pcl_msg) {
        ros::Time Begin_of_callback_image_time = ros::Time::now();

        sensor_msgs::PointCloud2 filtered_pcl;
        Cloud::Ptr tmp_pcl (new Cloud());
        Cloud::Ptr result  (new Cloud());
        pcl::fromROSMsg(*pcl_msg, *tmp_pcl);
        ros::Time begin = ros::Time::now();
        filter_crop_box(tmp_pcl, result);
        ros::Time end = ros::Time::now();
        std::cout << "Crop Time: " << end-begin << std::endl;


        begin = ros::Time::now();
        filterVoxelGrid(result, result);
        end = ros::Time::now();
        std::cout << "voxel time: " << end-begin << std::endl;


        begin = ros::Time::now();
        statistical_Outlair_Removal(result,result);
        end = ros::Time::now();
        std::cout << "Statistical Outlair Time: " << end-begin << std::endl;

        begin = ros::Time::now();
        filterHSV(result,result);
        end = ros::Time::now();
        std::cout << "HSV filter time: " << end-begin << std::endl;

        pcl::toROSMsg(*result, filtered_pcl);
        vis_pub.publish(filtered_pcl);

        ros::Time End_of_callback_image_time = ros::Time::now();
        std::cout << "total time: " << End_of_callback_image_time-Begin_of_callback_image_time << std::endl;

    }

private:
    void filter_crop_box(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
        pcl::CropBox<Point> cb;
        cb.setMin(minVal);
        cb.setMax(maxVal);
        cb.setInputCloud(pcl_msg);
        cb.filter(*filtered);
    }

    void statistical_Outlair_Removal(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
        pcl::StatisticalOutlierRemoval<Point> sor;
        sor.setInputCloud (pcl_msg);
        sor.setMeanK (10);
        sor.setStddevMulThresh (1.0);
        sor.filter (*filtered);
    }

    void filterVoxelGrid(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
        pcl::VoxelGrid<Point> sor;
        sor.setInputCloud(pcl_msg);
        sor.setLeafSize (voxelsize, voxelsize, voxelsize);
        sor.filter (*filtered);
    }

    void filterHSV(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
        cv::Mat RGBMat, HSVMat;
        if (pcl_msg->isOrganized()) {
            ROS_INFO("WUT");
            RGBMat = cv::Mat(pcl_msg->height, pcl_msg->width, CV_8UC3);

            if (!pcl_msg->empty()) {
                for (int h=0; h<RGBMat.rows; h++) {
                    for (int w=0; w<RGBMat.cols; w++) {
                        Point point = pcl_msg->at(w, h);

                        Eigen::Vector3i rgb = point.getRGBVector3i();

                        RGBMat.at<cv::Vec3b>(h,w)[0] = rgb[2];
                        RGBMat.at<cv::Vec3b>(h,w)[1] = rgb[1];
                        RGBMat.at<cv::Vec3b>(h,w)[2] = rgb[0];
                    }
                }
            }
            cv::cvtColor(RGBMat, HSVMat, CV_BGR2HSV);
            cv::inRange(HSVMat, lower, upper, RGBMat);

            cv::imshow("Display window", RGBMat);
            cv::waitKey(1);
        }
        ROS_INFO("pcl_msg size: %d RGBMat size: (%d, %d)", pcl_msg->size(), RGBMat.rows, RGBMat.cols);


    }

    cv::Scalar lower, upper;
    Eigen::Vector4f minVal, maxVal;
    int hmin, smin, vmin, hmax, smax, vmax;
    double voxelsize;
    ros::Subscriber pcl_sub;
    ros::Publisher vis_pub;
    ros::NodeHandle nh;

};

int main(int argc, char** argv){
    ros::init(argc, argv, "object_detection");
    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE);
//    cv::Mat RGBMat = cv::Mat::ones(480, 640, CV_8UC3);
//    cv::imshow("Display window", RGBMat);
//    cv::waitKey(0);
    object_detection od;
    ros::spin();

}
