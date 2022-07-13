#ifndef BINARY_POSE_ERROR_FACTOR_H
#define BINARY_POSE_ERROR_FACTOR_H
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <mutex>
#include <vector>

#include "../utility/utility.h"
#include "ceres/ceres.h"

/**
 * Binary_pose_error_facotr
 * Binary edge to constraint two SE3 poses
 * 二元边缘约束两个 SE3 位姿
 */
class Binary_pose_error_facotr : public ceres::SizedCostFunction<6, 7, 7> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<6, 7> base_t;

  /// \brief The number of residuals  se3参数化
  static const int kNumResiduals = 6;

  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 6, 6> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, 6, 7> jacobian_t;

  /// \brief The type of the Jacobian w.r.t. poses --
  /// \warning This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, 6, 6> jacobian0_t;

  Binary_pose_error_facotr(const Eigen::Isometry3d &_Pose0,
                           const Eigen::Isometry3d &_Pose1, double _t0,
                           double _t1, double weightScalar = 1);

  /// \brief Trivial destructor.
  virtual ~Binary_pose_error_facotr() {}
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                    double *residuals, double **jacobians,
                                    double **jacobiansMinimal) const;

 private:
  Eigen::Isometry3d Pose0_, Pose1_;  // T_wc0 T_wc1
  Eigen::Quaterniond dq_meas_;       // 两帧之间的 q
  Eigen::Vector3d dt_meas_;          // 两帧之间的 p
  // times, Not uesed now;
  double t0, t1;
  double weightScalar_;
  ///< information matrix and its square root
  mutable covariance_t covariance_;
  //< The information matrix for this error term.
  mutable information_t information_;
  //< The square root information matrix for this error term.
  mutable information_t squareRootInformation_;
};
#endif