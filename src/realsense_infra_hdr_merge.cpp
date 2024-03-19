
#include "realsense_infra_hdr_merge/realsense_infra_hdr_merge.hpp"

namespace rshdr {
  realsenseInfraHdrMerge::realsenseInfraHdrMerge(const rclcpp::NodeOptions& options) 
  : Node("realsense_infra_hdr_merge_node", options)
  {

    // Parameters
    saturationW_ = this->declare_parameter("saturation_weight",1.0f);
    exposureW_ = this->declare_parameter("exposure_weight",1.0f);
    contrastW_ = this->declare_parameter("contrast_weight",1.0f);
    // Create subscribers
    image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, "~/camera/image");
    info_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(this, "~/camera/info");
    metadata_sub_ = std::make_shared<message_filters::Subscriber<realsense2_camera_msgs::msg::Metadata>>(this, "~/camera/metadata");

    // Create the synchronizer with the policy
    sync_ = std::make_shared<Synchronizer<ExactSyncPolicy>>(ExactSyncPolicy(10), *image_sub_, *info_sub_, *metadata_sub_);
    sync_->registerCallback(std::bind(&realsenseInfraHdrMerge::callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    // Create the MergeMertens filter
    mergeMertens_ = cv::createMergeMertens();
    mergeMertens_->setSaturationWeight(saturationW_);
    mergeMertens_->setExposureWeight(exposureW_);
    mergeMertens_->setContrastWeight(contrastW_);
    // Create publishers
    ci_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("~/camera/camera_info", 2);
    im_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("~/camera/image_hdr_merge", 2);
    RCLCPP_INFO(this->get_logger(),"Initialized & subscribed Contrast Weight %f, Saturation Weight %f, Exposure Weight %f",
                                      contrastW_,saturationW_, exposureW_);
    images_.resize(2);  
  }

  int realsenseInfraHdrMerge::getSeqIDFromMetadataMsg(
    const realsense2_camera_msgs::msg::Metadata::ConstSharedPtr & metadata)
  {
    // Field name in json metadata
    constexpr char sequence_id_str[] = "\"sequence_id\":";
    constexpr size_t field_name_length =
      sizeof(sequence_id_str) / sizeof(sequence_id_str[0]);
    // Find the field
    const size_t sequence_id_start_location =
      metadata->json_data.find(sequence_id_str);
    // If the emitter mode is not found, return unknown and warn the user.
    if (sequence_id_start_location == metadata->json_data.npos) {
      constexpr int kPublishPeriodMs = 1000;
      auto & clk = *get_clock();
      RCLCPP_WARN_THROTTLE(
        get_logger(), clk, kPublishPeriodMs,
        "Realsense frame metadata did not contain \"sequence_id\". Merge will not work.");
      return -1;
    }
    // If it is found, parse the field.
    const size_t field_location = sequence_id_start_location + field_name_length - 1;
    const int sequence_id = atoi(&metadata->json_data[field_location]);
    return sequence_id;
  }

  int realsenseInfraHdrMerge::getSeqSizeFromMetadataMsg(
    const realsense2_camera_msgs::msg::Metadata::ConstSharedPtr & metadata)
  {
    // Field name in json metadata
    constexpr char sequence_size_str[] = "\"sequence_size\":";
    constexpr size_t field_name_length =
      sizeof(sequence_size_str) / sizeof(sequence_size_str[0]);
    // Find the field
    const size_t sequence_size_start_location =
      metadata->json_data.find(sequence_size_str);
    // If the emitter mode is not found, return unknown and warn the user.
    if (sequence_size_start_location == metadata->json_data.npos) {
      constexpr int kPublishPeriodMs = 1000;
      auto & clk = *get_clock();
      RCLCPP_WARN_THROTTLE(
        get_logger(), clk, kPublishPeriodMs,
        "Realsense frame metadata did not contain \"sequence_size\". Merge will not work.");
      return -1;
    }
    // If it is found, parse the field.
    const size_t field_location = sequence_size_start_location + field_name_length - 1;
    const int sequence_size = atoi(&metadata->json_data[field_location]);
    return sequence_size;
  }

  // The call back function 
  void realsenseInfraHdrMerge::callback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info,
                const realsense2_camera_msgs::msg::Metadata::ConstSharedPtr& camera_metadata)
  {
    int seqId = getSeqIDFromMetadataMsg(camera_metadata); // seq_id
    int seqSize  = getSeqSizeFromMetadataMsg(camera_metadata);  // total number in sequence

    if (seqId == -1 || seqSize == -1) return;    // Save image in vector
    if (seqId != prevSeqId_) {
      prevSeqId_ = seqId;
      inQueue_++;
    }
    images_[seqId] = cv_bridge::toCvCopy(image)->image;
    RCLCPP_DEBUG(this->get_logger(), "Received synchronized Image, CameraInfo, and CameraMetadata messages %d, (%dx%d)", seqId, images_[seqId].cols, images_[seqId].rows);
    // If all images are received, merge them
    if (inQueue_ == seqSize) {
      // Get the start time
      auto start_time = this->get_clock()->now();

      cv::Mat fusion, fusionNorm;
      mergeMertens_->process(images_, fusion);      // merge images
      fusion.convertTo(fusionNorm, CV_8UC1, 255.0); // normalize
      auto imbridge = cv_bridge::CvImage(image->header, sensor_msgs::image_encodings::MONO8, fusionNorm).toImageMsg();
      ci_publisher_->publish(*camera_info);
      im_publisher_->publish(*imbridge);
      // Get the end time
      auto end_time = this->get_clock()->now();
      auto duration = end_time - start_time;
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                           "Processed synchronized Image, CameraInfo, and CameraMetadata messages in %f seconds, %ld nanoseconds", 
                           duration.seconds(), duration.nanoseconds());
      inQueue_--;
    } else {
      RCLCPP_DEBUG(this->get_logger(), "Not processed synchronized Image, CameraInfo, and CameraMetadata messages %d", seqId);
    }
  }

} // namespace your_namespace

RCLCPP_COMPONENTS_REGISTER_NODE(rshdr::realsenseInfraHdrMerge)
/*json_data: '{"frame_number":196568,"clock_domain":"global_time","frame_timestamp":1710613353226.444824,"frame_counter":196568,"hw_timestamp":1840953149,"sensor_timestamp":1840953149,"actual_exposure":1,
"gain_level":16,"auto_exposure":0,"time_of_arrival":1710613353233,"backend_timestamp":0,"actual_fps":90,"frame_laser_power":0,"frame_laser_power_mode":0,"exposure_priority":1,"exposure_roi_left":106,
"exposure_roi_right":742,"exposure_roi_top":60,"exposure_roi_bottom":420,"frame_emitter_mode":0,"raw_frame_size":814080,"gpio_input_data":0,"sequence_name":0,"sequence_id":1,"sequence_size":2}'
---
*/