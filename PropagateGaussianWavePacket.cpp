#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>

using namespace Eigen;
using namespace std;

constexpr double PLANE[] = {0., 0., 1., 1.};
constexpr size_t GRID_SIZE[] = {16, 16};
constexpr double PACKET_SIZE[] = {0.1, 0.1};
constexpr double PACKET_K = 1000.;
constexpr double TIME_STEP = 0.001;
const double ALLOWED_ERROR = pow(10, -13);
constexpr size_t MAX_ORDER_OF_CHEBYSHEV_POLYNOMIAL = 100000;

constexpr size_t OPERATOR_SIZE = GRID_SIZE[0] * GRID_SIZE[1];
constexpr double PLANE_SIZE[] = {PLANE[2] - PLANE[0], PLANE[3] - PLANE[1]};
constexpr double MESH_STEP = PLANE_SIZE[0] / double(GRID_SIZE[0]);



complex<double> initialWaveUnnormalized(const double &x, const double &y) {
    if (x < PLANE[0] || x > PLANE[2] || y < PLANE[1] || y > PLANE[3]) {
        return 0.;
    } else {
        return exp(-pow(x - (PLANE[0] + PLANE[2]) / 2., 2) / 4. / pow(PACKET_SIZE[0], 2) - pow(y - (PLANE[0] + PLANE[2]) / 2., 2) / 4. / pow(PACKET_SIZE[1], 2) + complex<double>(0, 1) * PACKET_K * x);
    }
}

VectorXcd normalizeWave(VectorXcd &wave) {
    double factor = 1. / (wave.norm() * MESH_STEP);
    wave *= factor;
    return wave;
}

VectorXcd getDiscretizedInitialWave() {
    VectorXcd result(OPERATOR_SIZE);
    for (size_t j = 0; j < GRID_SIZE[1]; ++j) {
        for (size_t i = 0; i < GRID_SIZE[0]; ++i) {
            result[j * GRID_SIZE[1] + i] = initialWaveUnnormalized(i * MESH_STEP, j * MESH_STEP);
            cout << "pong" << endl;
        }
    }
    normalizeWave(result);
    return result;
}

double getPotential(const double &x, const double &y) {
    return 0.;
}

VectorXd flattenedRowHamiltionian(const long &i, const long &j) {
    VectorXd row(OPERATOR_SIZE);
    row = VectorXd::Constant(OPERATOR_SIZE, 0.0);
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

MatrixXcd getHamiltonian() {
    MatrixXcd H(OPERATOR_SIZE, OPERATOR_SIZE);
    for (size_t j = 0; j < GRID_SIZE[1]; ++j) {
        for (size_t i = 0; i < GRID_SIZE[0]; ++i) {
            H.block<1, OPERATOR_SIZE>(j * GRID_SIZE[1] + i, 0) = flattenedRowHamiltionian(i, j).transpose();
        }
    }
    return H;
}


const auto H = getHamiltonian();
const auto maxEntry = H.cwiseAbs().maxCoeff();
MatrixXcd getEvolutionOperatorForOneTimestep() {
    double z = TIME_STEP * maxEntry;
    auto B = H / maxEntry;
    MatrixXcd evolutionOperator;
    evolutionOperator = MatrixXcd::Zero(OPERATOR_SIZE, OPERATOR_SIZE);
    complex<double> jv = 1;
    size_t i = 1;
    vector<MatrixXcd> tildeTMatrices = {MatrixXcd::Identity(OPERATOR_SIZE, OPERATOR_SIZE), B * complex<double>(0., 1.)};
    while (abs(jv) > ALLOWED_ERROR && i <= MAX_ORDER_OF_CHEBYSHEV_POLYNOMIAL) {
        jv = cyl_bessel_j(i, z);
        evolutionOperator += jv * tildeTMatrices[0];
        auto nextTildeT = B * complex<double>(0., 2.) * tildeTMatrices[1] + tildeTMatrices[0];
        tildeTMatrices[0] = tildeTMatrices[1];
        tildeTMatrices[1] = nextTildeT;
        ++i;
    }
    evolutionOperator *= 2.0;
    evolutionOperator += MatrixXcd::Identity(OPERATOR_SIZE, OPERATOR_SIZE) * cyl_bessel_j(0, z);
    return evolutionOperator;
}



const auto evolutionOperator = getEvolutionOperatorForOneTimestep();
VectorXcd currentWave;

VectorXcd propagateWave() {
    currentWave = evolutionOperator * currentWave;
    normalizeWave(currentWave);
    return currentWave;
}

int main() {
    currentWave = getDiscretizedInitialWave();
    for (size_t i = 0; i < 1000; ++i) {
        propagateWave();
        cout << i << endl;
    }
    return 0;
}
