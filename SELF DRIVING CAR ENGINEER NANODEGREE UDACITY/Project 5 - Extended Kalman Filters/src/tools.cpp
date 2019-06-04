#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  VectorXd rmse(4);
    rmse << 0,0,0,0;

    if(estimations.size() == 0){
      std::cout << "ERROR - CalculateRMSE () - The estimations vector is empty" << std::endl;
      return rmse;
    }

    if(ground_truth.size() == 0){
      std::cout << "ERROR - CalculateRMSE () - The ground-truth vector is empty" << std::endl;
      return rmse;
    }

    unsigned int n = estimations.size();
    if(n != ground_truth.size()){
      std::cout << "ERROR - CalculateRMSE () - The ground-truth and estimations vectors must have the same size." << std::endl;
      return rmse;
    }

    for(unsigned int i=0; i < estimations.size(); ++i){
      VectorXd diff = estimations[i] - ground_truth[i];
      diff = diff.array()*diff.array();
      rmse += diff;
    }

    rmse = rmse / n;
    rmse = rmse.array().sqrt();
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3,4);

  if ( x_state.size() != 4 ) {
    std::cout << "ERROR - CalculateJacobian () - The state vector must have size 4." << std::endl;
    return Hj;
  }
    //State Parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    //Pre-compute a set of terms to avoid Repeated Calculation
    double c1 = px*px+py*py;
    double c2 = sqrt(c1);
    double c3 = (c1*c2);

    //Checking the Division by Zero
    if(fabs(c1) < 0.0001){
        std::cout << "ERROR - CalculateJacobian () - Division by Zero" << std::endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
          -(py/c1), (px/c1), 0, 0,
          py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}
