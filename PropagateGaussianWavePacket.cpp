#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std;

constexpr double PLANE[] = {0., 0., 1., 1.};
constexpr size_t GRID_SIZE[] = {64, 64};
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

size_t findProperApproximationOrder(double z) {
    size_t upperBound = MAX_ORDER_OF_CHEBYSHEV_POLYNOMIAL;
    size_t lowerBound = 0;
    size_t middlePoint = (upperBound + lowerBound) / 2;
    while (!(cyl_bessel_j(middlePoint, z) > ALLOWED_ERROR && cyl_bessel_j(middlePoint + 1, z) < ALLOWED_ERROR)) {
        if (cyl_bessel_j(middlePoint, z) > ALLOWED_ERROR) {
            lowerBound = middlePoint;
        } else {
            upperBound = middlePoint;
        }
        middlePoint = (lowerBound + upperBound) / 2;
    }
    return middlePoint + 1;
}


const auto H = getHamiltonian();
const auto maxEntry = H.cwiseAbs().maxCoeff();
MatrixXcd getEvolutionOperatorForOneTimestep() {
    auto begin = chrono::system_clock::now();
    double z = TIME_STEP * maxEntry;
    auto B = -H / maxEntry;
    MatrixXcd evolutionOperator;
    evolutionOperator = MatrixXcd::Zero(OPERATOR_SIZE, OPERATOR_SIZE);
    complex<double> jv = 1;
    vector<MatrixXcd> tildeTMatrices = {MatrixXcd::Identity(OPERATOR_SIZE, OPERATOR_SIZE), B * complex<double>(0., 1.)};
    auto properOrder = findProperApproximationOrder(z);
    for (size_t i = 1; i <= properOrder; ++i) {
        jv = cyl_bessel_j(i, z);
        size_t count = 0;
        for (size_t j = 0; j < OPERATOR_SIZE; ++j) {
            for (size_t k = 0; k < OPERATOR_SIZE; ++k) {
                if (tildeTMatrices[0](j, k) == 0.0) {
                    count++;
                }
            }
        }
        cout << OPERATOR_SIZE * OPERATOR_SIZE - count << endl;
        evolutionOperator += jv * tildeTMatrices[0];
        auto nextTildeT = B * complex<double>(0., 2.) * tildeTMatrices[1] + tildeTMatrices[0];
        tildeTMatrices[0] = tildeTMatrices[1];
        tildeTMatrices[1] = nextTildeT;
        cout << i << endl;
    }
    evolutionOperator *= 2.0;
    evolutionOperator += MatrixXcd::Identity(OPERATOR_SIZE, OPERATOR_SIZE) * cyl_bessel_j(0, z);
    auto end = chrono::system_clock::now();
    chrono::duration<double> timeCost = end - begin;
    cout << "init_minc : " << timeCost.count() << "s." << endl;
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
