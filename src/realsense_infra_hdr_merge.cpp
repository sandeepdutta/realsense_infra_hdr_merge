
#include "realsense_infra_hdr_merge/realsense_infra_hdr_merge.hpp"

namespace rshdr {
  realsenseInfraHdrMerge::realsenseInfraHdrMerge(const rclcpp::NodeOptions& options) 
  : Node("realsense_infra_hdr_merge_node", options)
  {

    // Parameters
    saturationW_ = this->declare_parameter("saturation_weight",1.0f);
    exposureW_ = this->declare_parameter("exposure_weight",1.0f);
    contrastW_ = this->declare_parameter("contrast_weight",1.0f);
    mergeDepth_ = this->declare_parameter("merge_depth",false);
    temporalFilterThreshold_ = this->declare_parameter("temporal_filter_threshold",100);
    temporalFilterFrames_ = this->declare_parameter("temporal_filter_frames",8);

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
    RCLCPP_INFO(this->get_logger(),"Initialized & subscribed Contrast Weight %f, Saturation Weight %f, Exposure Weight %f, Temporal Filter Threshold %d",
                                      contrastW_,saturationW_, exposureW_, temporalFilterThreshold_);
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

  // use MergeMertens to merge images and publish
  bool realsenseInfraHdrMerge::hdrMergeInfra(cv::Mat &fusionNorm) 
  {
    cv::Mat fusion;
    mergeMertens_->process(images_, fusion);      // merge images
    fusion.convertTo(fusionNorm, CV_8UC1, 255.0); // normalize
    return true;
  }

bool realsenseInfraHdrMerge::temporalFilter(cv::Mat& filtered_frame) {

    // Check if the input frames are of type CV_16U
    for (const auto& frame : images_) {
        if (frame.type() != CV_16U) {
            std::cerr << "Error: Input frames must be of type CV_16U." << std::endl;
            return false;
        }
    }

    // Convert the new frames to float for accurate averaging and add them to the buffer
    for (const auto& frame : images_) {
        cv::Mat frame_float;
        frame.convertTo(frame_float, CV_32F);
        frame_buffer_.push_back(frame_float);
    }
    // frame buffer size not reached yet
    if (frame_buffer_.size() < temporalFilterFrames_) {
        return false;
    }

    // Remove the oldest frames if the buffer exceeds the number of frames
    while (frame_buffer_.size() > temporalFilterFrames_) {
        frame_buffer_.pop_front();
    }

    // Initialize the accumulator for the average
    cv::Mat avg_frame = cv::Mat::zeros(images_[0].size(), CV_32F);

    // Create a mask to track valid pixels across all frames
    cv::Mat valid_mask = cv::Mat::ones(images_[0].size(), CV_8U); // Start with all pixels valid
  
    // Update the mask based on the threshold
    for (const auto& frame : frame_buffer_) {
        cv::Mat current_mask;
        cv::compare(frame, temporalFilterThreshold_, current_mask, cv::CMP_GT);
        valid_mask &= current_mask;
    }

    cv::Mat valid_mask_float ;
    valid_mask.convertTo(valid_mask_float, CV_32F);

    // Accumulate the valid frames
    for (const auto& frame : frame_buffer_) {
        avg_frame += frame.mul(valid_mask_float);
    }
    avg_frame /= temporalFilterFrames_;
    
    // Convert the result back to 16-bit unsigned integer
    avg_frame.convertTo(filtered_frame, CV_16U);

    return true;
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
      bool merged = false;
      // Get the start time
      auto start_time = this->get_clock()->now();
      cv::Mat fused ;
      if (mergeDepth_) {
        merged = temporalFilter(fused);
      } else {
        merged = hdrMergeInfra(fused);
      }
      if (merged) {
        sensor_msgs::msg::Image::ConstSharedPtr imbridge;
        if (mergeDepth_) {
          imbridge = cv_bridge::CvImage(image->header,sensor_msgs::image_encodings::MONO16, fused).toImageMsg();
        } else {
          imbridge = cv_bridge::CvImage(image->header,sensor_msgs::image_encodings::MONO8, fused).toImageMsg();
        }
        ci_publisher_->publish(*camera_info);
        im_publisher_->publish(*imbridge);
      }
      /*
      cv::Mat fusion, fusionNorm;
      mergeMertens_->process(images_, fusion);      // merge images
      fusion.convertTo(fusionNorm, CV_8UC1, 255.0); // normalize
      auto imbridge = cv_bridge::CvImage( sensor_msgs::image_encodings::MONO8, fusionNorm).toImageMsg();
      ci_publisher_->publish(*camera_info);
      im_publisher_->publish(*imbridge);
      */
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