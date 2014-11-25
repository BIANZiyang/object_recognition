#include <cstdio>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <boost/thread.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
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
#include <pcl/filters/statistical_outlier_removal.h>

typedef pcl::PCLPointCloud2 Cloud2;
typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> Cloud;
typedef pcl::PointXYZHSV PointHSV;
typedef pcl::PointCloud<PointHSV> CloudHSV;

class object_detection{
public:
    object_detection() :
        it_(nh_)
    {
        loadParams();

        pcl_sub_ = nh_.subscribe("/snapshot/pcl", 1, &object_detection::pointCloudCB, this);
        img_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1, &object_detection::imageCB, this);
        img_pub_ = it_.advertise("/object_recognition/filtered_image",1);
        pcl_tf_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/object_detection/transformed", 1);

        hsvRanges_.resize(5);
        currentCloudPtr_ = Cloud::Ptr(new Cloud);

        cv::namedWindow("HSV filter", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("Depth filter", CV_WINDOW_AUTOSIZE);
        selectedHsvRange_ = 0;
        lastHsvRange_ = selectedHsvRange_;
        setupHsvTrackbars();
        uiThread = boost::thread(&object_detection::asyncImshow, this);
    }

    ~object_detection() {
        uiThread.interrupt();
        uiThread.join();
    }

    void imageCB(const sensor_msgs::ImageConstPtr& img_msg) {
        cv_bridge::CvImagePtr cvPtr;
        try {
            cvPtr = cv_bridge::toCvCopy(img_msg, "bgr8");
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        currentImagePtr_ = cvPtr;
    }

    void pointCloudCB(const sensor_msgs::PointCloud2ConstPtr& pclMsg) {

        pcl::fromROSMsg(*pclMsg, *currentCloudPtr_);
        tf::StampedTransform transform;
        try {
            tf_sub_.lookupTransform("camera_rgb_optical", "robot_base", ros::Time(0), transform);
        } catch (tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
        }
        pcl_ros::transformPointCloud(*currentCloudPtr_, *currentCloudPtr_, transform);
        sensor_msgs::PointCloud2 msgOut;
        pcl::toROSMsg(*currentCloudPtr_, msgOut);
        pcl_tf_pub_.publish(msgOut);

    }

    void detect() {

        pcl::CropBox<Point> cb;
        std::vector<int> indices;
        cb.setMin(cbMin_);
        cb.setMax(cbMax_);
        cb.setInputCloud(currentCloudPtr_);
        cb.filter(indices);

        int rows = currentCloudPtr_->height;
        int cols = currentCloudPtr_->width;
        if(rows > 0 && cols > 0) {
            cv::Mat depthMask = cv::Mat::zeros(rows, cols, CV_8UC1);
            for(size_t i = 0; i < indices.size(); ++i) {
                depthMask.at<char>(indices[i]) = 255;
            }

            //Temporary
            cv::Mat RGBMat(rows, cols, CV_8UC3);
            for(size_t i = 0; i < rows*cols; ++i) {
                Eigen::Vector3i rgb = currentCloudPtr_->at(i).getRGBVector3i();

                RGBMat.at<cv::Vec3b>(i)[0] = rgb[2];
                RGBMat.at<cv::Vec3b>(i)[1] = rgb[1];
                RGBMat.at<cv::Vec3b>(i)[2] = rgb[0];
            }
            cv::imshow("HSV filter", RGBMat);
            cv::imshow("Depth filter", depthMask);
        }
    }

private:
    class hsvRange {
    public:

        hsvRange() :
            hmin(min[0]), smin(min[1]), vmin(min[2]),
            hmax(max[0]), smax(max[1]), vmax(max[2])
        {
        }

        hsvRange(const hsvRange& other) :
            hmin(min[0]), smin(min[1]), vmin(min[2]),
            hmax(max[0]), smax(max[1]), vmax(max[2])
        {
            min = other.min;
            max = other.max;
        }

        hsvRange& operator=(const hsvRange& other) {
            min = other.min;
            max = other.max;
        }

        template <typename T> void setValues(const T& hmin, const T& smin, const T& vmin,
                                             const T& hmax, const T& smax, const T& vmax) {
            this->hmin = hmin;
            this->smin = smin;
            this->vmin = vmin;
            this->hmax = hmax;
            this->smax = smax;
            this->vmax = vmax;
        }

        cv::Scalar min;
        cv::Scalar max;

        double& hmin;
        double& smin;
        double& vmin;
        double& hmax;
        double& smax;
        double& vmax;
    };



    void loadParams(){
        getParam("object_detection/crop/wMin", cbMin_[0], -0.3);    //width
        getParam("object_detection/crop/hMin", cbMin_[1], 0.01);    //height
        getParam("object_detection/crop/dMin", cbMin_[2], 0.1);     //depth
        getParam("object_detection/crop/wMax", cbMax_[0], 0.3);     //width
        getParam("object_detection/crop/hMax", cbMax_[1], 0.07);    //height
        getParam("object_detection/crop/dMax", cbMax_[2], 3);       //depth

        getParam("object_detection/voxel/leafsize", voxelsize_, 0.005);

        std::string hsvParamName("object_detection/hsv");
        for(size_t i = 0; i < hsvRanges_.size(); ++i) {
            std::stringstream ss; ss << hsvParamName << i;
            getParam(ss.str() + "/hmin", hsvRanges_[i].hmin, 0);
            getParam(ss.str() + "/smin", hsvRanges_[i].smin, 0);
            getParam(ss.str() + "/vmin", hsvRanges_[i].vmin, 0);
            getParam(ss.str() + "/hmax", hsvRanges_[i].hmax, 180);
            getParam(ss.str() + "/smax", hsvRanges_[i].smax, 255);
            getParam(ss.str() + "/vmax", hsvRanges_[i].vmax, 255);
        }
    }

    template <typename T1, typename T2> bool getParam(const std::string paramName, T1& variable, const T2& standardValue) {
        if(nh_.hasParam(paramName)) {
            nh_.getParam(paramName, variable);
            return true;
        }
        variable = standardValue;
        return false;
    }

    template <typename T> bool getParam(const std::string paramName, float& variable, const T& standardValue) {
        if(nh_.hasParam(paramName)) {
            double temp;
            nh_.getParam(paramName, temp);
            variable = temp;
            return true;
        }
        variable = standardValue;
        return false;
    }

    void asyncImshow() {
        ros::Rate rate(30);
        while(1) {
            try {
                boost::this_thread::interruption_point();
                updateHsvTrackbars();
                cv::waitKey(1);
                rate.sleep();
            } catch(boost::thread_interrupted&){
                return;
            }
        }
    }

    void setupHsvTrackbars() {
        cv::namedWindow("HSVTrackbars",CV_WINDOW_NORMAL);
        cv::createTrackbar("Index", "HSVTrackbars", &selectedHsvRange_, hsvRanges_.size()-1);
        cv::createTrackbar("Hmin", "HSVTrackbars", NULL, 180);
        cv::createTrackbar("Hmax", "HSVTrackbars", NULL, 180);
        cv::createTrackbar("Smin", "HSVTrackbars", NULL, 255);
        cv::createTrackbar("Smax", "HSVTrackbars", NULL, 255);
        cv::createTrackbar("Vmin", "HSVTrackbars", NULL, 255);
        cv::createTrackbar("Vmax", "HSVTrackbars", NULL, 255);
    }

    void updateHsvTrackbars() {
        if(lastHsvRange_ != selectedHsvRange_) {
            lastHsvRange_ = selectedHsvRange_;
            cv::setTrackbarPos("Hmin", "HSVTrackbars", hsvRanges_[selectedHsvRange_].hmin);
            cv::setTrackbarPos("Hmax", "HSVTrackbars", hsvRanges_[selectedHsvRange_].hmax);
            cv::setTrackbarPos("Smin", "HSVTrackbars", hsvRanges_[selectedHsvRange_].smin);
            cv::setTrackbarPos("Smax", "HSVTrackbars", hsvRanges_[selectedHsvRange_].smax);
            cv::setTrackbarPos("Vmin", "HSVTrackbars", hsvRanges_[selectedHsvRange_].vmin);
            cv::setTrackbarPos("Vmax", "HSVTrackbars", hsvRanges_[selectedHsvRange_].vmax);
        }

        hsvRanges_[selectedHsvRange_].hmin = cv::getTrackbarPos("Hmin", "HSVTrackbars");
        hsvRanges_[selectedHsvRange_].hmax = cv::getTrackbarPos("Hmax", "HSVTrackbars");
        hsvRanges_[selectedHsvRange_].smin = cv::getTrackbarPos("Smin", "HSVTrackbars");
        hsvRanges_[selectedHsvRange_].smax = cv::getTrackbarPos("Smax", "HSVTrackbars");
        hsvRanges_[selectedHsvRange_].vmin = cv::getTrackbarPos("Vmin", "HSVTrackbars");
        hsvRanges_[selectedHsvRange_].vmax = cv::getTrackbarPos("Vmax", "HSVTrackbars");
    }

//    void filter_crop_box(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
//        pcl::CropBox<Point> cb;
//        cb.setMin(cbMin_);
//        cb.setMax(cbMax_);
//        cb.setInputCloud(pcl_msg);
//        cb.filter(*filtered);
//    }

//    void statistical_Outlair_Removal(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
//        pcl::StatisticalOutlierRemoval<Point> sor;
//        sor.setInputCloud (pcl_msg);
//        sor.setMeanK (10);
//        sor.setStddevMulThresh (1.0);
//        sor.filter (*filtered);
//    }

//    void filterVoxelGrid(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
//        pcl::VoxelGrid<Point> sor;
//        sor.setInputCloud(pcl_msg);
//        sor.setLeafSize (voxelsize_, voxelsize_, voxelsize_);
//        sor.filter (*filtered);
//    }

//    void filterHSV(Cloud::Ptr& pcl_msg, Cloud::Ptr& filtered){
//        cv::Mat RGBMat(pcl_msg->height, pcl_msg->width, CV_8UC3);
//        cv::Mat HSVMat;
//        cv::Mat resMat(pcl_msg->height, pcl_msg->width, CV_8U);

//        if (pcl_msg->isOrganized()) {
//            if (!pcl_msg->empty()) {
//                for (int h=0; h<RGBMat.rows; h++) {
//                    for (int w=0; w<RGBMat.cols; w++) {
//                        Point point = pcl_msg->at(w, h);
//                        Eigen::Vector3i rgb = point.getRGBVector3i();

//                        RGBMat.at<cv::Vec3b>(h,w)[0] = rgb[2];
//                        RGBMat.at<cv::Vec3b>(h,w)[1] = rgb[1];
//                        RGBMat.at<cv::Vec3b>(h,w)[2] = rgb[0];
//                    }
//                }
//            }
//        }
//        //cv::medianBlur(RGBMat, RGBMat, 7);
//        cv::GaussianBlur(RGBMat, RGBMat, cv::Size(15,15), 0, 0);
//        cv::cvtColor(RGBMat, HSVMat, CV_BGR2HSV);
//        cv::inRange(HSVMat, lower1, upper1, resMat);
//        cv::medianBlur(resMat, resMat, 9);

//        std::vector<int> indices;
//        for(int i = 0; i < pcl_msg->size(); i++) {
//            if(!resMat.at<unsigned char>(0, i)) indices.push_back(i);
//        }

//        *filtered = Cloud(*pcl_msg, indices);

//        cv::imshow("Display window", RGBMat);
//    }

    ros::NodeHandle nh_;
    ros::Subscriber pcl_sub_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber img_sub_;
    tf::TransformListener tf_sub_;
    image_transport::Publisher img_pub_;
    ros::Publisher pcl_tf_pub_;

    boost::thread uiThread;

    double voxelsize_;
    int selectedHsvRange_;
    int lastHsvRange_;
    int hmin_, smin_, vmin_, hmax_, smax_, vmax_;
    Eigen::Vector4f cbMin_, cbMax_;
    std::vector<hsvRange> hsvRanges_;

    Cloud::Ptr currentCloudPtr_;
    cv_bridge::CvImagePtr currentImagePtr_;
};



int main(int argc, char** argv){
    ros::init(argc, argv, "object_detection");
    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE);
    cv::namedWindow( "Mask1 window", CV_WINDOW_AUTOSIZE);
    cv::namedWindow( "Mask2 window", CV_WINDOW_AUTOSIZE);
    cv::namedWindow( "Combined", CV_WINDOW_AUTOSIZE);

    object_detection od;

    ros::Rate rate(0.2);
    while(ros::ok()) {
        ros::spinOnce();
        od.detect();
        rate.sleep();
    }

}

