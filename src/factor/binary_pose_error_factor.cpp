#include "binary_pose_error_factor.hpp"

Binary_pose_error_facotr::Binary_pose_error_facotr(
    const Eigen::Isometry3d &_Pose0, const Eigen::Isometry3d &_Pose1,
    double _t0, double _t1, double weightScalar)
    : Pose0_(_Pose0),
      Pose1_(_Pose1),
      t0(_t0),
      t1(_t1),
      weightScalar_(weightScalar) {
  // q_c0c1
  dq_meas_ =
      Eigen::Quaterniond(Pose0_.rotation().transpose() * Pose1_.rotation());
  dt_meas_ = Pose1_.translation() - Pose0_.translation();
}

bool Binary_pose_error_facotr::Evaluate(double const *const *parameters,
                                        double *residuals,
                                        double **jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool Binary_pose_error_facotr::EvaluateWithMinimalJacobians(
    double const *const *parameters, double *residuals, double **jacobians,
    double **jacobiansMinimal) const {
  // T_WI0
  Eigen::Vector3d t_hat0 =
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Q_hat0(parameters[0][6], parameters[0][3],
                            parameters[0][4], parameters[0][5]);
  // T_WI1
  Eigen::Vector3d t_hat1 =
      Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Q_hat1(parameters[1][6], parameters[1][3],
                            parameters[1][4], parameters[1][5]);

  Eigen::Matrix<double, 6, 1> error;
  Eigen::Quaterniond invdQ_meas = dq_meas_.inverse(); // q 逆 --> q_c1c0
  Eigen::Quaterniond dq_hat = Q_hat0.inverse() * Q_hat1;  // q_I0I1
  Eigen::Quaterniond ddq = invdQ_meas * dq_hat; // 1 : 理论上应该是单位四元数

  // std::cout<<"dq_hat: "<<dq_hat.coeffs().transpose()<<std::endl;
  // std::cout<<"dq_meas_: "<<dq_meas_.coeffs().transpose()<<std::endl;
  // std::cout<<"invdQ_meas: "<<invdQ_meas.coeffs().transpose()<<std::endl;
  // std::cout<<"ddq: "<<ddq.coeffs().transpose()<<std::endl;
  /*
  dq_hat: 0 0 0 1
  dq_meas_: 0 0 0 1
  invdQ_meas: -0 -0 -0  1
  ddq: 0 0 0 1
  */
  const Eigen::Vector3d dtheta = 2 * ddq.coeffs().head<3>();

  Eigen::Vector3d dt_hat = (t_hat1 - t_hat0);
  error.head<3>() = dt_hat - dt_meas_;
  // std::cout << "dt_hat: " << dt_hat.transpose() << ", dt_meas_: " << dt_meas_.transpose()
  //           << ", err_p: " << (dt_hat - dt_meas_).transpose() <<std::endl; // 0 0 0
  error.tail<3>() = dtheta;

  squareRootInformation_.setIdentity();
  squareRootInformation_ = weightScalar_ * squareRootInformation_;  // Weighted

  // weight it
  Eigen::Map<Eigen::Matrix<double, 6, 1>> weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  // calculate jacobians
  if (jacobians != NULL) {
    // step: 1.对第i帧的位姿 pbi,qbi求导
    if (jacobians[0] != NULL) {
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> jacobian0_min;
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian0(
          jacobians[0]);

      jacobian0_min.setIdentity();
      jacobian0_min.topLeftCorner(3, 3) = -Eigen::Matrix3d::Identity();
      //?: 不太理解这里是怎么求导的 - q_c1c0 * q_I0I1
      jacobian0_min.bottomRightCorner(3, 3) =
          -(Utility::quatPlus(invdQ_meas) * Utility::quatOplus(dq_hat))
               .topLeftCorner(3, 3);

      jacobian0_min = squareRootInformation_ * jacobian0_min;

      jacobian0 << jacobian0_min, Eigen::Matrix<double, 6, 1>::Zero();  // lift

      if (jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>
            map_jacobian0_min(jacobiansMinimal[0]);
        map_jacobian0_min = jacobian0_min;
      }
    }
    // step: 2.对第j帧的位姿 pbi,qbi求导
    if (jacobians[1] != NULL) {
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> jacobian1_min;
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian1(
          jacobians[1]);

      jacobian1_min.setIdentity();
      jacobian1_min.bottomRightCorner(3, 3) =
          Utility::quatPlus(ddq).topLeftCorner(3, 3);

      jacobian1_min = squareRootInformation_ * jacobian1_min;

      jacobian1 << jacobian1_min, Eigen::Matrix<double, 6, 1>::Zero();  // lift

      if (jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>
            map_jacobian1_min(jacobiansMinimal[1]);
        map_jacobian1_min = jacobian1_min;
      }
    }
  }

  return true;
}
