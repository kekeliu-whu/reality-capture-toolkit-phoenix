#include <ros.h>
#include "interface/lio.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lixel_lio_node");
  ros::NodeHandle nh("lixel_lio_node");

  std::string algorithm_config_filename;
  if (!nh.getParam("algorithm_config_filename", algorithm_config_filename))
  {
    std::cerr << "Please specify filename of configuration!" << std::endl;
    return -1;
  }

  lixel::ParametersReader parameters_reader(algorithm_config_filename);
  lixel::LioParameters parameters;
  parameters_reader.getParameters(parameters);

  lixel::LioCore lio_instance(parameters);

  lixel_ros::Subscriber subscriber(nh, &lio_instance, parameters);
  lixel_ros::Publisher publisher(nh);
  lio_instance.setLioResultCallback(
      std::bind(&lixel_ros::Publisher::lioResultCallback, &publisher, std::placeholders::_1));

  lio_instance.start();

  ros::spin();
  return 0;
}