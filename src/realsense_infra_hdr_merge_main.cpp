#include <rclcpp/rclcpp.hpp>

#include "realsense_infra_hdr_merge/realsense_infra_hdr_merge.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  rclcpp::spin(std::make_shared<rshdr::realsenseInfraHdrMerge>(node_options));
  rclcpp::shutdown();
  return 0;
}
