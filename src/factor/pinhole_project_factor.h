#ifndef DEPTH_PROJECT_FACTOR_H
#define DEPTH_PROJECT_FACTOR_H
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <mutex>
#include <vector>

#include "../utility/utility.h"
#include "ceres/ceres.h"

/**
 * note:解析求导（Analytic Differentiation）
 * https://blog.csdn.net/supengufo/article/details/106462460?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-5-106462460-blog-78651579.pc_relevant_multi_platform_whitelistv2&spm=1001.2101.3001.4242.4&utm_relevant_index=8
 */
class PinholeProjectFactor
    : public ceres::SizedCostFunction<2, /* num of residual */
                                      7, /* parameter of pose */
                                      7, /* parameter of pose */
                                      7, 1> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<2, 7, 7> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 2;

  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 2, 2> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, 2, 7> jacobian_t;

  /// \brief The type of the Jacobian w.r.t. poses --
  /// \warning This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, 2, 6> jacobian0_t;

  PinholeProjectFactor() = delete;
  // 构造函数
  PinholeProjectFactor(const Eigen::Vector3d &uv_C0,
                       const Eigen::Vector3d &uv_C1);

  /// \brief Trivial destructor.
  virtual ~PinholeProjectFactor() {}

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                    double *residuals, double **jacobians,
                                    double **jacobiansMinimal) const;

 private:
  Eigen::Vector3d C0uv;
  Eigen::Vector3d C1uv;

  // information matrix and its square root
  ///< The information matrix for this error term.
  mutable information_t information_;
  ///< The square root information matrix for this error term.
  mutable information_t squareRootInformation_;
};

#endif