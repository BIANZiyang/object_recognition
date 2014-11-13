// Copyed from Hand_tracker
#include <cstdio>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

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

        sensor_msgs::PointCloud2 filtered_pcl;
        filterHSV(pcl_msg, filtered_pcl);
        vis_pub.publish(filtered_pcl);
    }

private:
    void filterHSV(const sensor_msgs::PointCloud2ConstPtr& pcl_msg, sensor_msgs::PointCloud2& filtered){
        CloudHSV::Ptr input(new pcl::PointCloud<PointHSV>);
        CloudHSV::Ptr output(new pcl::PointCloud<PointHSV>);
        Cloud2 tmp_pcl;
        pcl_conversions::toPCL(*pcl_msg, tmp_pcl);
        pcl::fromPCLPointCloud2(tmp_pcl, *input);

//        pcl::PassThrough<PointHSV> pass;
//        pass.setInputCloud(input);
//        pass.setFilterFieldName ("x");
//        pass.setFilterLimits (hmin, hmax);
//        pass.filter (*output);
//        pass.setInputCloud(output);
//        pass.setFilterFieldName ("y");
//        pass.setFilterLimits (smin, smax);
//        pass.filter(*output);
//        pass.setInputCloud(output);
//        pass.setFilterFieldName ("z");
//        pass.setFilterLimits (vmin, vmax);
//        pass.filter(*output);

        pcl::toROSMsg(*output, filtered);
    }

    float hmin, smin, vmin, hmax, smax, vmax;
    ros::Subscriber pcl_sub;
    ros::Publisher vis_pub;
    ros::NodeHandle nh;

};

int main(int argc, char** argv){
    ros::init(argc, argv, "object_detection");
    object_detection od;
    ros::spin();

}
