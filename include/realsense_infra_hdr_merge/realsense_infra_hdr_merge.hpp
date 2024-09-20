#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <deque>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <realsense2_camera_msgs/msg/metadata.hpp>
#include "rclcpp_components/register_node_macro.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/exact_time.h"
#include "message_filters/synchronizer.h"
#include "sensor_msgs/msg/camera_info.hpp" 

using namespace message_filters;

namespace rshdr {

// Define ExactTime sync policy
typedef sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo,realsense2_camera_msgs::msg::Metadata> ExactSyncPolicy;

class realsenseInfraHdrMerge : public rclcpp::Node
{
public:
  realsenseInfraHdrMerge(const rclcpp::NodeOptions& options) ;
  
private:
    int getSeqIDFromMetadataMsg(const realsense2_camera_msgs::msg::Metadata::ConstSharedPtr & metadata);
    int getSeqSizeFromMetadataMsg(const realsense2_camera_msgs::msg::Metadata::ConstSharedPtr & metadata);
    bool hdrMergeInfra(cv::Mat &fusionNorm);
    bool temporalFilter(cv::Mat& filtered_frame);
    void callback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                  const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info,
                  const realsense2_camera_msgs::msg::Metadata::ConstSharedPtr& camera_metadata);

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> info_sub_;
  std::shared_ptr<message_filters::Subscriber<realsense2_camera_msgs::msg::Metadata>> metadata_sub_;
  std::shared_ptr<Synchronizer<ExactSyncPolicy>> sync_;
  std::vector<cv::Mat> images_;
  std::deque<cv::Mat> frame_buffer_;
  cv::Ptr<cv::MergeMertens> mergeMertens_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr ci_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr im_publisher_;
  int prevSeqId_ {1};
  int inQueue_ {0};
  // Parameters
  float saturationW_  {1.0};
  float exposureW_    {1.0};
  float contrastW_    {1.0};
  bool mergeDepth_    {false};
  int temporalFilterThreshold_ {100};
  unsigned int temporalFilterFrames_   {8};
};

} 