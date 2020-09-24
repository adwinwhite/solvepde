#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>

using namespace Eigen;
using namespace std;

constexpr double PLANE[] = {0, 0, 1, 1};
constexpr size_t GRID_SIZE[] = {16, 16};
constexpr double PACKET_SIZE[] = {0.1, 0.1};
constexpr double PACKET_K = 1000;
constexpr double TIME_STEP = 0.001;
constexpr double ALLOWED_ERROR = 0.0;
constexpr size_t MAX_ORDER_OF_CHEBYSHEV_POLYNOMIAL = 100000;

constexpr size_t OPERATOR_SIZE = GRID_SIZE[0] * GRID_SIZE[1];
constexpr double PLANE_SIZE[] = {PLANE[2] - PLANE[0], PLANE[3] - PLANE[1]};
constexpr double MESH_STEP = PLANE_SIZE[0] / double(GRID_SIZE[0]);



complex<double> initialWaveUnnormalized(const double &x, const double &y) {
    if (x < PLANE[0] || x > PLANE[2] || y < PLANE[1] || y > PLANE[3]) {
        return 0;
    } else {
        return exp(-pow(x - (PLANE[0] + PLANE[2]) / 2., 2) / 4. / pow(PACKET_SIZE[0], 2) - pow(y - (PLANE[0] + PLANE[2]) / 2., 2) / 4. / pow(PACKET_SIZE[1], 2) + complex<double>(0, 1) * PACKET_K * x);
    }
}

double getInitialNormalizationFactor() {
    double integral = 0;
    for (size_t j = 0; j < GRID_SIZE[1]; ++j) {
        for (size_t i = 0; j < GRID_SIZE[0]; ++i) {
            integral += pow(abs(initialWaveUnnormalized(i * MESH_STEP, j *MESH_STEP)), 2);
        }
    }
    integral *= pow(MESH_STEP, 2);
    return sqrt(1/ integral);
}

complex<double> initialWaveNormalized(const double &x, const double &y) {
    return initialWaveUnnormalized(x, y) * getInitialNormalizationFactor();
}

VectorXcd getDiscretizedInitialWave() {
    VectorXcd result(OPERATOR_SIZE);
    for (size_t j = 0; j < GRID_SIZE[1]; ++j) {
        for (size_t i = 0; i < GRID_SIZE[0]; ++i) {
            result[j * GRID_SIZE[1] + i] = initialWaveNormalized(i * MESH_STEP, j * MESH_STEP);
        }
    }
    return result;
}

double getPotential(const double &x, const double &y) {
    return 0;
}

VectorXd flattenedRowHamiltionian(const long &i, const long &j) {
    VectorXd row(OPERATOR_SIZE, 0.0);
    row(j * GRID_SIZE[1] + i) = -4;
    if (i + 1 < GRID_SIZE[0]) {
        row(j * GRID_SIZE[1] + i + 1) = 1;
    }
    if (i - 1 >= 0) {
        row(j * GRID_SIZE[1] + i - 1) = 1;
    }
    if (j + 1 < GRID_SIZE[1]) {
        row((j + 1) * GRID_SIZE[1] + i) = 1;
    }
    if (j - 1 >= 0) {
        row((j - 1) * GRID_SIZE[1] + i) = 1;
    }
    row = row / pow(MESH_STEP, 2);
    row((j * GRID_SIZE[1] + i)) += getPotential(i * MESH_STEP, j * MESH_STEP);
    return row;
}

MatrixXd getHamiltonian() {
    MatrixXd H(OPERATOR_SIZE, OPERATOR_SIZE);
    for (size_t j = 0; j < GRID_SIZE[1]; ++j) {
        for (size_t i = 0; i < GRID_SIZE[0]; ++i) {
            H.block<1, OPERATOR_SIZE>(j * GRID_SIZE[1] + i, 0) = flattenedRowHamiltionian(i, j).transpose();
        }
    }
    return H;
}

vector<MatrixXcd> tildeTMatrices;
MatrixXcd getTildeTMatrix(const size_t &order, const MatrixXcd &B) {
    if (order < tildeTMatrices.size()) {
        return tildeTMatrices[order];
    }
    if (order == 0) {
        auto mat = MatrixXcd::Identity(OPERATOR_SIZE, OPERATOR_SIZE);
        tildeTMatrices.push_back(mat);
        return mat;
    } else if (order == 1) {
        auto mat = B * complex<double>(0, 1);
        tildeTMatrices.push_back(mat);
        return mat;
    } else {
        auto mat = B * complex<double>(0, 2) * getTildeTMatrix(order - 1, B) + getTildeTMatrix(order - 2, B);
        tildeTMatrices.push_back(mat);
        return mat;
    }
}

auto H = getHamiltonian();
auto maxEntry = H.cwiseAbs().maxCoeff();
MatrixXcd getEvolutionOperatorForOneTimestep() {
    double z = -TIME_STEP * maxEntry;
    auto B = H / maxEntry;
    auto evolutionOperator = MatrixXcd::Zero(OPERATOR_SIZE, OPERATOR_SIZE);
    complex<double> jv = 1;
    size_t i = 1;
    while (abs(jv) > ALLOWED_ERROR || i <= MAX_ORDER_OF_CHEBYSHEV_POLYNOMIAL) {
        jv = cyl_bessel_j(i, z);
        evolutionOperator += jv * getTildeTMatrix(i, B);
        ++i;
    }
    evolutionOperator *= 2;
    evolutionOperator += MatrixXcd::Identity(OPERATOR_SIZE, OPERATOR_SIZE) * cyl_bessel_j(0, z);
    return evolutionOperator;
}

VectorXcd normalizeWave(VectorXcd &wave) {
    double factor = 1 / (wave.norm() * MESH_STEP);
    wave *= factor;
    return wave;
}

auto evolutionOperator = getEvolutionOperatorForOneTimestep();
auto currentWave = getDiscretizedInitialWave();

VectorXcd propagateWave() {
    currentWave = normalizeWave(evolutionOperator * currentWave);
    return current_wave;
}
