#include "project_error.h"

#include "pose_local_parameterization.h"
ProjectError::ProjectError(const Eigen::Vector3d &uv_C0) : C0uv(uv_C0) {}

bool ProjectError::Evaluate(double const *const *parameters, double *residuals,
                            double **jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool ProjectError::EvaluateWithMinimalJacobians(
    double const *const *parameters, double *residuals, double **jacobians,
    double **jacobiansMinimal) const {
  // T_WC
  Eigen::Vector3d t_WC(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Q_WC(parameters[0][6], parameters[0][3], parameters[0][4],
                          parameters[0][5]);

  // Wp
  Eigen::Vector3d Wp(parameters[1][0], parameters[1][1], parameters[1][2]);

  Eigen::Matrix3d R_WC = Q_WC.toRotationMatrix();
  Eigen::Vector3d Cp = R_WC.transpose() * (Wp - t_WC); // 相机坐标系下的3D坐标
  Eigen::Matrix<double, 2, 1> error;

  double inv_z = 1 / Cp(2);
  Eigen::Vector2d hat_C0uv(Cp(0) * inv_z, Cp(1) * inv_z);
  // step: 1.视觉误差对重投影3D点fc求导
  Eigen::Matrix<double, 2, 3> H;
  H << 1, 0, -Cp(0) * inv_z, 0, 1, -Cp(1) * inv_z;
  H *= inv_z;

  error = hat_C0uv - C0uv.head<2>(); // 重投影误差
  squareRootInformation_.setIdentity();
  // squareRootInformation_ = weightScalar_* squareRootInformation_; //Weighted

  // weight it
  Eigen::Map<Eigen::Matrix<double, 2, 1>> weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  // calculate jacobians
  // step: 2 fc 对各状态量求导
  // note: 重投影误差对3D点求导参考第二版十四讲 7.7.3小节（P.186）
  // note: 或参考 https://blog.csdn.net/qq_42518956/article/details/107192217
  if (jacobians != NULL) {
    // step: 2.1. fc 对重投影帧 p q求导
    if (jacobians[0] != NULL) {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jacobian0_min;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian0(
          jacobians[0]);

      Eigen::Matrix<double, 3, 6> tmp;
      tmp.setIdentity();
      // 对 p 求导
      // ?: 为什么会有这一项
      tmp.topLeftCorner(3, 3) = R_WC.transpose();
      // 对 q 求导
      tmp.topRightCorner(3, 3) =
          -Utility::skewSymmetric(R_WC.transpose() * (Wp - t_WC));

      jacobian0_min = - H * tmp; // 链式法则合并结果

      jacobian0 << squareRootInformation_ * jacobian0_min,
          Eigen::Matrix<double, 2, 1>::Zero();

      if (jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
            map_jacobian0_min(jacobiansMinimal[0]);
        map_jacobian0_min = squareRootInformation_ * jacobian0_min;
      }
    }
    // step: 2.2. fc 对投影帧 p 求导
    if (jacobians[1] != NULL) {
      Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobian1_min;
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian1(
          jacobians[1]);

      Eigen::Matrix<double, 3, 3> tmp;

      tmp = R_WC.transpose();
      jacobian1_min = H * tmp;

      jacobian1 = squareRootInformation_ * jacobian1_min;

      if (jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
            map_jacobian1_min(jacobiansMinimal[1]);
        map_jacobian1_min = squareRootInformation_ * jacobian1_min;
      }
    }
  }
  return true;
}