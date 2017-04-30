#include "ICPSimulation.h"
#include <ceres/ceres.h>
#include <vector>

class ICPErr : public ceres::SizedCostFunction<3, 6> {
public:
    ICPErr(Eigen::Vector3d& pi, Eigen::Vector3d &pj,
                 Eigen::Matrix<double, 3, 3> &information);
    virtual ~ICPErr() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

public:
    Eigen::Vector3d Pi;
    Eigen::Vector3d Pj;
    Eigen::Matrix<double, 3, 3> sqrt_information_;
};


bool ICPErr::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(parameters[0]);
    Sophus::SE3d T = Sophus::SE3d::exp(lie);

    //std::cout << T.matrix3x4() << std::endl;

    auto Pj_ = T * Pi;
    Eigen::Vector3d err = Pj - Pj_;

    err = sqrt_information_ * err;

    residuals[0] = err(0);
    residuals[1] = err(1);
    residuals[2] = err(2);

    Eigen::Matrix<double, 3, 6> Jac = Eigen::Matrix<double, 3, 6>::Zero();
    Jac.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    Jac.block<3, 3>(0, 3) = Sophus::SO3d::hat(Pj_);
    int k = 0;
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 6; ++j) {
            if(k >= 18)
                return false;
            if(jacobians) {
                if(jacobians[0])
                    jacobians[0][k] = Jac(i, j);
            }
            k++;
        }
    }

    //printf("jacobian ok!\n");

    return true;

}

ICPErr::ICPErr(Eigen::Vector3d& pi, Eigen::Vector3d &pj,
                           Eigen::Matrix<double, 3, 3>& information) :  Pi(pi), Pj(pj) {

    //printf("index = %d\n", index++);
    Eigen::LLT<Eigen::Matrix<double, 3, 3>> llt(information);
    sqrt_information_ = llt.matrixL();
}


class CERES_EXPORT SE3Parameterization : public ceres::LocalParameterization {
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

bool SE3Parameterization::ComputeJacobian(const double *x, double *jacobian) const {
    ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
    return true;
}

bool SE3Parameterization::Plus(const double* x,
                  const double* delta,
                  double* x_plus_delta) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

    Sophus::SE3d T = Sophus::SE3d::exp(lie);
    Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_lie);
    Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

    for(int i = 0; i < 6; ++i) x_plus_delta[i] = x_plus_delta_lie(i, 0);

    return true;

}

/* ############################################################################################
 * ############################################################################################
 */


ICPSimulation::ICPSimulation(Sophus::SE3d &se3, Eigen::Matrix<double, 3, 3>& Var, int npts) {
    real_ = se3;
    information_ = Var.inverse();
    npt_ = npts;
}

void ICPSimulation::start() {

    double se3[6];
	memset(se3, 0, 6 * sizeof(double));

    ceres::Problem problem;
	std::vector<Eigen::Vector3d> Pi, Pj;
	Eigen::Matrix3d var = information_.inverse();
	sampleUniformMeans<double, 3>(-10, 10, Pi, npt_);
	for(size_t i = 0; i < Pi.size(); ++i) {
		auto Pj_ = real_ * Pi[i];
		Pj.push_back(oneSampleGauss<double, 3>(Pj_, var));
	}


    for(size_t i = 0; i < Pi.size(); ++i) {
        ceres::CostFunction * costFun = new ICPErr(Pi[i], Pj[i], information_);
        problem.AddResidualBlock(costFun, new ceres::HuberLoss(0.5), se3);
    }

    problem.SetParameterization(se3, new SE3Parameterization());

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = true;
    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
	//printf("%lf, %lf, %lf, %lf, %lf, %lf\n", se3[0], se3[1], se3[2], se3[3], se3[4], se3[5]);
	std::cout << "real = \n" << real_.matrix3x4() << std::endl;
	Eigen::Map<Eigen::Matrix<double, 6, 1> > se3lie(se3);
	std::cout << "esitmate = \n" << Sophus::SE3d::exp(se3lie).matrix3x4() << std::endl;
}

ICPSimulation::~ICPSimulation() {}