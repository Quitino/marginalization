#pragma once

#include <ceres/ceres.h>
#include <pthread.h>

#include <cstdlib>
#include <unordered_map>

#include "../utility/utility.h"

/*
 * Used to store any kinds of factors information
 * 用于存储各种因子信息
 */
struct ResidualBlockInfo {
  // 构造函数
  ResidualBlockInfo(ceres::CostFunction *_cost_function,      // 输入：（约束）pinholeProjectFactor
                    ceres::LossFunction *_loss_function,      // NULL
                    std::vector<double *> _parameter_blocks,  // 输出：待优化遍变量param_T_WI0, param_T_WI1, param_T_IC,param_rho.at(i)
                    std::vector<int> _drop_set)               // std::vector<int>{0, 3}  边缘化掉 T_WI0?
      : cost_function(_cost_function),
        loss_function(_loss_function),
        parameter_blocks(_parameter_blocks),  //
        drop_set(_drop_set) {}

  void Evaluate();

  ceres::CostFunction *cost_function;
  ceres::LossFunction *loss_function;
  std::vector<double *> parameter_blocks;  // 状态变量数据
  std::vector<int> drop_set;               // 待边缘化的状态变量id

  double **raw_jacobians;                  // 雅克比矩阵
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      jacobians;
  Eigen::VectorXd residuals;               // 残差 IMU:15X1 视觉2X1

  int localSize(int size) { return size == 7 ? 6 : size; }
};

struct ThreadsStruct {
  std::vector<ResidualBlockInfo *> sub_factors;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  std::unordered_map<long, int> parameter_block_size;  // global size
  std::unordered_map<long, int> parameter_block_idx;   // local size
};

class MarginalizationInfo {
 public:
  ~MarginalizationInfo();
  int localSize(int size) const;
  int globalSize(int size) const;
  //添加参差块相关信息（优化变量，待marg的变量）
  void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
  //计算每个残差对应的雅克比，并更新parameter_block_data
  void preMarginalize();
  //pos为所有变量维度，ｍ为需要marg掉的变量，ｎ为需要保留的变量
  void marginalize();
  std::vector<double *> getParameterBlocks(
      std::unordered_map<long, double *> &addr_shift);
  // factors involing to parametre need be marginalized
  // 所有观测项
  std::vector<ResidualBlockInfo *> factors;
  int m, n;  // m为要边缘化的变量个数，n为要保留下来的变量个数
  // global size, map address to size
  // <优化变量内存地址,localSize=变量参数化的维度 7 >
  std::unordered_map<long, int> parameter_block_size;
  int sum_block_size;
  // <待边缘化的优化变量内存地址,在parameter_block_size中的id>
  // local size, map address
  std::unordered_map<long, int> parameter_block_idx;  
  // map address
  // <优化变量内存地址,数据>
  std::unordered_map<long, double *> parameter_block_data;  
  std::vector<int> keep_block_size;  // global size
  std::vector<int> keep_block_idx;   // local size
  std::vector<double *> keep_block_data;

  Eigen::MatrixXd linearized_jacobians;  // 由先验信息恢复的Jacobian
  Eigen::VectorXd linearized_residuals;  // 由先验信息恢复的残差
  const double eps = 1e-8;
};

class MarginalizationFactor : public ceres::CostFunction {
 public:
  MarginalizationFactor(MarginalizationInfo *_marginalization_info);

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                    double *residuals, double **jacobians,
                                    double **jacobiansMinimal) const;

  MarginalizationInfo *marginalization_info;
};
