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

#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>

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
        hmin=0;
        hmax=120;
        smin=0;
        smax=255;
        vmin=0;
        vmax=255;

    }
    void imageCb(const sensor_msgs::PointCloud2ConstPtr& pcl_msg) {

//        filterHSV(pcl_msg, filtered_pcl);

//        filterVoxelGrid(pcl_msg, filtered_pcl);
//        sensor_msgs::PointCloud2ConstPtr temp;
//        temp.reset(&filtered_pcl);
//        statistical_Outlair_Removal(temp, result);
        sensor_msgs::PointCloud2 filtered_pcl;
        Cloud::Ptr tmp_pcl (new Cloud());
        Cloud::Ptr result  (new Cloud());
        pcl::fromROSMsg(*pcl_msg, *tmp_pcl);
        filterVoxelGrid(tmp_pcl,result);
        statistical_Outlair_Removal(result,result);
        pcl::toROSMsg(*result, filtered_pcl);
        vis_pub.publish(filtered_pcl);
    }

private:
    void statistical_Outlair_Removal(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
//        Cloud::Ptr output (new Cloud ());
//        Cloud::Ptr tmp_pcl (new Cloud());
//        pcl::fromROSMsg(*pcl_msg, *tmp_pcl);


        pcl::StatisticalOutlierRemoval<Point> sor;
        sor.setInputCloud (pcl_msg);
        sor.setMeanK (30);
        sor.setStddevMulThresh (1.0);
        sor.filter (*filtered);
//        pcl::toROSMsg(*output, filtered);

    }

    void filterVoxelGrid(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
//        Cloud::Ptr output (new Cloud ());
//        Cloud::Ptr tmp_pcl (new Cloud());
//        pcl::fromROSMsg(*pcl_msg, *tmp_pcl);
        pcl::VoxelGrid<Point> sor;
        sor.setInputCloud(pcl_msg);
        sor.setLeafSize (0.005f, 0.005f, 0.005f);
        sor.filter (*filtered);

//        pcl::toROSMsg(*output, filtered);

    }

    void filterHSV(const sensor_msgs::PointCloud2ConstPtr& pcl_msg, sensor_msgs::PointCloud2& filtered){
        Cloud::Ptr input(new Cloud);
        Cloud::Ptr output(new Cloud);
        Cloud2 tmp_pcl;
        pcl_conversions::toPCL(*pcl_msg, tmp_pcl);
        pcl::fromPCLPointCloud2(tmp_pcl, *input);

        cv::Mat RGBMat, HSVMat;
        if (input->isOrganized()) {
            RGBMat = cv::Mat(input->height, input->width, CV_8UC3);

            if (!input->empty()) {
                for (int h=0; h<RGBMat.rows; h++) {
                    for (int w=0; w<RGBMat.cols; w++) {
                        Point point = input->at(w, h);

                        Eigen::Vector3i rgb = point.getRGBVector3i();

                        RGBMat.at<cv::Vec3b>(h,w)[0] = rgb[2];
                        RGBMat.at<cv::Vec3b>(h,w)[1] = rgb[1];
                        RGBMat.at<cv::Vec3b>(h,w)[2] = rgb[0];
                    }
                }
            }
        }

        ROS_INFO("pcl_msg size: %d RGBMat size: (%d, %d)", input->size(), RGBMat.rows, RGBMat.cols);


        cv::Scalar lower(207/2-20, 45*2.55, 70*2.55);
        cv::Scalar upper(207/2+20, 55*2.55, 80*2.55);
        cv::cvtColor(RGBMat, HSVMat, CV_BGR2HSV);
        cv::inRange(HSVMat, lower, upper, RGBMat);

        cv::imshow("Display window", RGBMat);
        cv::waitKey(1);



//        pcl::toROSMsg(*output, filtered);

    }

    float hmin, smin, vmin, hmax, smax, vmax;
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
