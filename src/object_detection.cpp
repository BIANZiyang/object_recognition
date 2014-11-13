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
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

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
    object_detection() :
        _it(nh)
    {
        //pcl_sub = nh.subscribe("/camera/depth_registered/points", 1, &object_detection::pointCloudCB, this);
        //vis_pub = nh.advertise<sensor_msgs::PointCloud2>("/object_recognition/deteciton", 1);
        img_sub = _it.subscribe("/camera/rgb/image_rect_color", 1, &object_detection::imageCB, this);
        img_pub = _it.advertise("/object_recognition/filtered_image",1);
        load_from_launchfile();
        cv::namedWindow("HSVTrackbars",CV_WINDOW_NORMAL);
        cv::createTrackbar("Hmin1","HSVTrackbars",&hmin1,180);
        cv::createTrackbar("Hmax1","HSVTrackbars",&hmax1,180);
        cv::createTrackbar("Smin1","HSVTrackbars",&smin1,255);
        cv::createTrackbar("Smax1","HSVTrackbars",&smax1,255);
        cv::createTrackbar("Vmin1","HSVTrackbars",&vmin1,255);
        cv::createTrackbar("Vmax1","HSVTrackbars",&vmax1,255);
        cv::createTrackbar("Hmin2","HSVTrackbars",&hmin2,180);
        cv::createTrackbar("Hmax2","HSVTrackbars",&hmax2,180);
        cv::createTrackbar("Smin2","HSVTrackbars",&smin2,255);
        cv::createTrackbar("Smax2","HSVTrackbars",&smax2,255);
        cv::createTrackbar("Vmin2","HSVTrackbars",&vmin2,255);
        cv::createTrackbar("Vmax2","HSVTrackbars",&vmax2,255);
        cv::imshow("HSVTrackbars",1);
    }
    void load_from_launchfile(){
        voxelsize=0.005;
        hmin1=0;
        hmax1=120;
        smin1=0;
        smax1=255;
        vmin1=0;
        vmax1=255;
        double temp_leafsize;
        double depth = 0.3, height = 0.3, width = 0.3, heightOffset = 0.3, depthOffset = 0.1;
        if(nh.hasParam("object_detection/leafsize")) {nh.getParam("object_detection/leafsize", temp_leafsize); voxelsize=temp_leafsize;}
        if(nh.hasParam("object_detection/depth")) {nh.getParam("object_detection/depth", depth);}
        if(nh.hasParam("object_detection/height")) {nh.getParam("object_detection/height", height);}
        if(nh.hasParam("object_detection/width")) {nh.getParam("object_detection/width", width);}
        if(nh.hasParam("object_detection/heightOffset")) {nh.getParam("object_detection/heightOffset", heightOffset);}
        if(nh.hasParam("object_detection/depthOffset")) {nh.getParam("object_detection/depthOffset", depthOffset);}

        if(nh.hasParam("object_detection/hmin1")) {nh.getParam("object_detection/hmin1", hmin1);}
        if(nh.hasParam("object_detection/hmax1")) {nh.getParam("object_detection/hmax1", hmax1);}
        if(nh.hasParam("object_detection/smin1")) {nh.getParam("object_detection/smin1", smin1);}
        if(nh.hasParam("object_detection/smax1")) {nh.getParam("object_detection/smax1", smax1);}
        if(nh.hasParam("object_detection/vmin1")) {nh.getParam("object_detection/vmin1", vmin1);}
        if(nh.hasParam("object_detection/vmax1")) {nh.getParam("object_detection/vmax1", vmax1);}
        if(nh.hasParam("object_detection/hmin2")) {nh.getParam("object_detection/hmin2", hmin2);}
        if(nh.hasParam("object_detection/hmax2")) {nh.getParam("object_detection/hmax2", hmax2);}
        if(nh.hasParam("object_detection/smin2")) {nh.getParam("object_detection/smin2", smin2);}
        if(nh.hasParam("object_detection/smax2")) {nh.getParam("object_detection/smax2", smax2);}
        if(nh.hasParam("object_detection/vmin2")) {nh.getParam("object_detection/vmin2", vmin2);}
        if(nh.hasParam("object_detection/vmax2")) {nh.getParam("object_detection/vmax2", vmax2);}


        //Order of params: width, height, depth
        minVal[0] = -width/2;
        minVal[1] = -height/2 - heightOffset;
        minVal[2] = -depth/2 + depthOffset;
        maxVal[0] = width/2;
        maxVal[1] = height/2 - heightOffset;
        maxVal[2] = depth/2 + depthOffset;
        lower1[0] = hmin1;
        lower1[1] = smin1;
        lower1[2] = vmin1;
        upper1[0] = hmax1;
        upper1[1] = smax1;
        upper1[2] = vmax1;
        lower2[0] = hmin2;
        lower2[1] = smin2;
        lower2[2] = vmin2;
        upper2[0] = hmax2;
        upper2[1] = smax2;
        upper2[2] = vmax2;

    }

    void imageCB(const sensor_msgs::ImageConstPtr& img_msg) {
        lower1[0] = hmin1;
        lower1[1] = smin1;
        lower1[2] = vmin1;
        upper1[0] = hmax1;
        upper1[1] = smax1;
        upper1[2] = vmax1;
        lower2[0] = hmin2;
        lower2[1] = smin2;
        lower2[2] = vmin2;
        upper2[0] = hmax2;
        upper2[1] = smax2;
        upper2[2] = vmax2;
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat blured, resMat1, resMat2, combined,result;
        cv::GaussianBlur(cv_ptr->image, blured, cv::Size(15,15), 0, 0);
        cv::imshow("Display window", blured);
        cv::cvtColor(blured, resMat1, CV_BGR2HSV);
        cv::cvtColor(blured, resMat2, CV_BGR2HSV);
        cv::cvtColor(blured,result,CV_BGR2HSV);
        cv::inRange(resMat1, lower1, upper1, resMat1);
        cv::inRange(resMat2, lower2, upper2, resMat2);
        resMat2 = 255-resMat2;
        cv::imshow("Mask1 window", resMat1);
        cv::imshow("Mask2 window", resMat2);
        combined = resMat1 & resMat2;

        cv::medianBlur(combined, combined, 9);
        combined = 255-combined;
        cv::imshow("Combined", combined);

        cv::Mat locations;
        cv::findNonZero(combined, locations);

        if(locations.rows>=0){

            //cv::Rect rec= cv::boundingRect(locations);
            cv::Rect rec(245,165,150,150);
            cv::Mat recImage = cv::Mat(result,rec);
            cv_ptr->image=recImage;


            img_pub.publish(cv_ptr->toImageMsg());
        }



    }

    void pointCloudCB(const sensor_msgs::PointCloud2ConstPtr& pcl_msg) {
        lower1[0] = hmin1;
        lower1[1] = smin1;
        lower1[2] = vmin1;
        upper1[0] = hmax1;
        upper1[1] = smax1;
        upper1[2] = vmax1;
        ros::Time Begin_of_callback_image_time = ros::Time::now();

        sensor_msgs::PointCloud2 filtered_pcl;
        Cloud::Ptr tmp_pcl (new Cloud());
        Cloud::Ptr result  (new Cloud());
        pcl::fromROSMsg(*pcl_msg, *tmp_pcl);

        filterHSV(tmp_pcl, result);

        ros::Time begin = ros::Time::now();
        filter_crop_box(result, result);
        ros::Time end = ros::Time::now();
        std::cout << "Crop Time: " << end-begin << std::endl;


//        begin = ros::Time::now();
//        filterVoxelGrid(result, result);
//        end = ros::Time::now();
//        std::cout << "voxel time: " << end-begin << std::endl;


//        begin = ros::Time::now();
//        statistical_Outlair_Removal(result,result);
//        end = ros::Time::now();
//        std::cout << "Statistical Outlair Time: " << end-begin << std::endl;

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
        cv::Mat RGBMat(pcl_msg->height, pcl_msg->width, CV_8UC3);
        cv::Mat HSVMat;
        cv::Mat resMat(pcl_msg->height, pcl_msg->width, CV_8U);

        if (pcl_msg->isOrganized()) {
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
        }
        //cv::medianBlur(RGBMat, RGBMat, 7);
        cv::GaussianBlur(RGBMat, RGBMat, cv::Size(15,15), 0, 0);
        cv::cvtColor(RGBMat, HSVMat, CV_BGR2HSV);
        cv::inRange(HSVMat, lower1, upper1, resMat);
        cv::medianBlur(resMat, resMat, 9);

        std::vector<int> indices;
        for(int i = 0; i < pcl_msg->size(); i++) {
            if(!resMat.at<unsigned char>(0, i)) indices.push_back(i);
        }

        *filtered = Cloud(*pcl_msg, indices);

        cv::imshow("Display window", RGBMat);
    }

    cv::Scalar lower1, upper1, lower2, upper2;
    Eigen::Vector4f minVal, maxVal;
    int hmin1, smin1, vmin1, hmax1, smax1, vmax1;
    int hmin2, smin2, vmin2, hmax2, smax2, vmax2;
    double voxelsize;
    ros::NodeHandle nh;
    image_transport::ImageTransport _it;
    image_transport::Subscriber img_sub;
    image_transport::Publisher img_pub;
    ros::Subscriber pcl_sub;
    ros::Publisher vis_pub;

};

int main(int argc, char** argv){
    ros::init(argc, argv, "object_detection");
    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE);
    cv::namedWindow( "Mask1 window", CV_WINDOW_AUTOSIZE);
    cv::namedWindow( "Mask2 window", CV_WINDOW_AUTOSIZE);
    cv::namedWindow( "Combined", CV_WINDOW_AUTOSIZE);

    object_detection od;
    ros::Rate rate(30);
    while(ros::ok()) {
        ros::spinOnce();
        rate.sleep();

        cv::waitKey(1);

    }

    ros::spin();

}
