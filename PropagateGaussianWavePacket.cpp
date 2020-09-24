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
    for (size_t j = 0; i < GRID_SIZE[1]; ++i) {
        for (size_t i = 0; j < GRID_SIZE[0]; ++j) {
            integral += pow(abs(initialWaveUnnormalized(i * MESH_STEP, j *MESH_STEP)), 2);
        }
    }
    integral *= MESH_STEP;
    return sqrt(1/ integral);
}

complex<double> initialWaveNormalized(const double &x, const double &y) {
    return initialWaveUnnormalized(x, y) * getInitialNormalizationFactor();
}

VectorXd getDiscretizedInitialWave() {
    VectorXd result(OPERATOR_SIZE);
    for (size_t j = 0; i < GRID_SIZE[1]; ++i) {
        for (size_t i = 0; i < GRID_SIZE[0]; ++j) {
            result[j * GRID_SIZE[1] + i] = initialWaveNormalized((i * MESH_STEP, j * MESH_STEP));
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

vector<MatrixXcd> getEvolutionOperator()
