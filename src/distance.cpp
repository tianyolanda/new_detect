#include "ros/ros.h"
#include "detect_pkg/OBJINFO.h"

double dis=0;
detect_pkg::OBJINFO obj;
void receivedistance(const detect_pkg::OBJINFO &msg)
{
 	dis = msg.distance1;
	ROS_INFO("distance is %f", dis);
}

int main(int argc, char** argv)
{
	ros::init(argc,argv,"dist");
	ros::NodeHandle n; 
        ros::Subscriber sub=n.subscribe("obj_info",1000,receivedistance);
        ROS_INFO("mdzz");
        ros::spin();
	return 0;
}

