//
// Created by lancelot on 3/15/17.
//

#ifndef PNP_PNPSIMULATION_H
#define PNP_PNPSIMULATION_H

#include <boost/math/distributions.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random.hpp>

#include <g2o/stuff/sampler.h>
#include <g2o/core/factory.h>

#include <g2o/stuff/sampler.h>

#include <Eigen/Dense>
#include <sophus/se3.hpp>


template<typename T, int N>
void sampleGauss(const Eigen::Matrix<T, N, 1>& mean, //!< in
                 const Eigen::Matrix<T, N, N>& var,  //!< in
                 std::vector<Eigen::Matrix<T, N, 1>>& vec, //!< out
                 int num = 50) {
    g2o::GaussianSampler<Eigen::Matrix<T, N, 1>, Eigen::Matrix<T, N, N>> gaussSampler;
    gaussSampler.setDistribution(var);
    for (int i = 0; i < num; ++i) {
        Eigen::Matrix<T, N, 1> v = mean + gaussSampler.generateSample();
        vec.push_back(v);
    }
}

template <typename T, int N>
Eigen::Matrix<T, N, 1> oneSampleGauss(const Eigen::Matrix<T, N, 1>& mean,
                 const Eigen::Matrix<T, N, N>& var) {
    g2o::GaussianSampler<Eigen::Matrix<T, N, 1>, Eigen::Matrix<T, N, N>> gaussSampler;
    gaussSampler.setDistribution(var);
    return mean + gaussSampler.generateSample();
}


template <typename T, int N>
void sampleUniformMeans(T start, T end, std::vector<Eigen::Matrix<T, N, 1>>& vec, int num = 50) {
    static boost::mt19937 rng(static_cast<unsigned>(std::time(0)));
    boost::uniform_real<T> uni_dist(start, end);
    Eigen::Matrix<T, N, 1> mean;
    for (int i = 0; i < num; ++i) {
        for(int dim = 0; dim < N; dim++)
            mean(dim) = uni_dist(rng);
        vec.push_back(mean);
    }
}

class ICPSimulation {
public:
    ICPSimulation(Sophus::SE3d& se3, Eigen::Matrix<double, 3, 3>& Var, int npt = 50);
    ~ICPSimulation();

    void start();

private:
    Sophus::SE3d real_;
    Eigen::Matrix3d information_;
    int npt_;
};


#endif //PNP_PNPSIMULATION_H
