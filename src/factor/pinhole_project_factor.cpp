#include "pinhole_project_factor.h"
PinholeProjectFactor::PinholeProjectFactor(const Eigen::Vector3d &uv_C0,
                                           const Eigen::Vector3d &uv_C1)
    : C0uv(uv_C0), C1uv(uv_C1) {}

bool PinholeProjectFactor::Evaluate(double const *const *parameters,
                                    double *residuals,
                                    double **jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool PinholeProjectFactor::EvaluateWithMinimalJacobians(
    double const *const *parameters, double *residuals, double **jacobians,
    double **jacobiansMinimal) const {
  // T_WI0
  Eigen::Vector3d t_WI0(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Q_WI0(parameters[0][6], parameters[0][3], parameters[0][4],
                           parameters[0][5]);

  // T_WI1
  Eigen::Vector3d t_WI1(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Q_WI1(parameters[1][6], parameters[1][3], parameters[1][4],
                           parameters[1][5]);

  // T_IC
  Eigen::Vector3d t_IC(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond Q_IC(parameters[2][6], parameters[2][3], parameters[2][4],
                          parameters[2][5]);

  // rho
  double inv_dep = parameters[3][0];

  // note:VIO中视觉约束Jacobian求导
  // https://blog.csdn.net/iwanderu/article/details/104729332
  // note:VINS-MONO ProjectionFactor代码分析及公式推导
  // https://www.cnblogs.com/glxin/p/11990551.html

  Eigen::Vector3d C0p = C0uv / inv_dep;     //第i帧相机坐标系下的3D坐标
  Eigen::Vector3d I0p = Q_IC * C0p + t_IC;  //第i帧IMU坐标系下的3D坐标
  Eigen::Vector3d Wp = Q_WI0 * I0p + t_WI0; //世界坐标系下的3D坐标
  Eigen::Vector3d I1p = Q_WI1.inverse() * (Wp - t_WI1); //第j帧imu坐标系下的3D坐标
  Eigen::Vector3d C1p = Q_IC.inverse() * (I1p - t_IC);  //第j帧相机坐标系下的3D坐标

  Eigen::Matrix<double, 2, 1> error; // 残差构建

  double inv_z = 1 / C1p(2);
  // 重投影点 c1p
  Eigen::Vector2d hat_C1uv(C1p(0) * inv_z, C1p(1) * inv_z);
  
  // 因为根据链式法则，对Jacobian的计算可以分成：
  // step: 1 视觉残差对重投影3D点fcj求导  2*3
  Eigen::Matrix<double, 2, 3> H;
  H << 1, 0, -C1p(0) * inv_z, 0, 1, -C1p(1) * inv_z;
  H *= inv_z;

  error = hat_C1uv - C1uv.head<2>();// 重投影误差
  squareRootInformation_.setIdentity();
  // squareRootInformation_ = weightScalar_* squareRootInformation_; //Weighted
  // weight it
  Eigen::Map<Eigen::Matrix<double, 2, 1>> weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  Eigen::Matrix3d R_WI0 = Q_WI0.toRotationMatrix();
  Eigen::Matrix3d R_WI1 = Q_WI1.toRotationMatrix();
  Eigen::Matrix3d R_IC = Q_IC.toRotationMatrix();

  // calculate jacobians
  // step: 2 fcj 对各状态量求导
  if (jacobians != NULL) {
    // step: 2.1.对第i帧的位姿 pbi,qbi求导      2X7的矩阵 最后一项是 0
    if (jacobians[0] != NULL) {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jacobian0_min;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian0(jacobians[0]);

      Eigen::Matrix<double, 3, 6> tmp;
      tmp.setIdentity();
      // 对p_bi(VINS中的b系为IMU系，注意和这里的I做对应)求导
      tmp.topLeftCorner(3, 3) = R_IC.transpose() * R_WI1.transpose();
      // 对q_bi)求导
      tmp.topRightCorner(3, 3) = -R_IC.transpose() * R_WI1.transpose() * R_WI0 *
                                 Utility::skewSymmetric(I0p);

      jacobian0_min = H * tmp; // 链式法则合并
      jacobian0_min = squareRootInformation_ * jacobian0_min;
      // 转为 2X7的矩阵 最后一项是0
      jacobian0 << jacobian0_min, Eigen::Matrix<double, 2, 1>::Zero();  // lift

      if (jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
            map_jacobian0_min(jacobiansMinimal[0]);
        map_jacobian0_min = jacobian0_min;
      }
    }
     // step: 2.2.对第j帧的位姿 p_bj,q_bj求导
    if (jacobians[1] != NULL) {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jacobian1_min;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian1(
          jacobians[1]);

      Eigen::Matrix<double, 3, 6> tmp;

      tmp.setIdentity();
      // 对p_bj求导
      tmp.topLeftCorner(3, 3) = -R_IC.transpose() * R_WI1.transpose();
      // 对q_bj求导
      tmp.bottomRightCorner(3, 3) =
          R_IC.transpose() * Utility::skewSymmetric(I1p);

      jacobian1_min = H * tmp; // 链式法则合并
      jacobian1_min = squareRootInformation_ * jacobian1_min;
      // 转为 2X7的矩阵 最后一项是0
      jacobian1 << jacobian1_min, Eigen::Matrix<double, 2, 1>::Zero();  // lift
      // ? 这里是在干嘛
      if (jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
            map_jacobian1_min(jacobiansMinimal[1]);
        map_jacobian1_min = jacobian1_min;
      }
    }
    // step: 2.3.对相机到IMU的外参 p_bc,q_bc (qic,tic)求导
    if (jacobians[2] != NULL) {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jacobian2_min;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian2(
          jacobians[2]);

      Eigen::Matrix<double, 3, 6> tmp;

      tmp.setIdentity();
      // 对p_bc求导
      tmp.topLeftCorner(3, 3) =
          R_IC.transpose() *
          (R_WI1.transpose() * R_WI0 - Eigen::Matrix3d::Identity());
      // 对q_bc求导
      Eigen::Matrix3d tmp_r =
          R_IC.transpose() * R_WI1.transpose() * R_WI0 * R_IC;
      tmp.bottomRightCorner(3, 3) =
          -tmp_r * Utility::skewSymmetric(C0p) +
          Utility::skewSymmetric(tmp_r * C0p) +
          Utility::skewSymmetric(
              R_IC.transpose() *
              (R_WI1.transpose() * (R_WI0 * t_IC + t_WI0 - t_WI1) - t_IC));

      jacobian2_min = H * tmp;// 链式法则合并
      jacobian2_min = squareRootInformation_ * jacobian2_min;
      // 转为 2X7的矩阵 最后一项是0
      jacobian2 << jacobian2_min, Eigen::Matrix<double, 2, 1>::Zero();  // lift

      if (jacobiansMinimal != NULL && jacobiansMinimal[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
            map_jacobian2_min(jacobiansMinimal[2]);
        map_jacobian2_min = jacobian2_min;
      }
    }
    // step: 3 对逆深度求导
    if (jacobians[3] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian3(jacobians[3]);
      jacobian3 = -H * R_IC.transpose() * R_WI1.transpose() * R_WI0 * R_IC *
                  C0uv / (inv_dep * inv_dep);
      jacobian3 = squareRootInformation_ * jacobian3;

      if (jacobiansMinimal != NULL && jacobiansMinimal[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> map_jacobian3_min(
            jacobiansMinimal[3]);
        map_jacobian3_min = jacobian3;
      }
    }
  }

  return true;
}
