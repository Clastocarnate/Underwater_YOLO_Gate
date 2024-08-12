#include <iostream>
#include <cmath>
using namespace std;
#include <Eigen/Core>
#include <Eigen/Geometry>

int main (int argc, char** argv){
    
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    cout << rotation_matrix << endl;
    Eigen::AngleAxisd rotation_vector(M_PI/4, Eigen::Vector3d (0,0,1));
    cout .precision(3);
    cout << rotation_vector.matrix();
    rotation_matrix = rotation_vector.toRotationMatrix();
    Eigen::Vector3d v(1,0,0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation is: " << v_rotated.transpose()<<endl;
    return 0;
}