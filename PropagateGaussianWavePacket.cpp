#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <algorithm>
//#include "matplotlibcpp.h"

using namespace Eigen;
using namespace std;

constexpr double PLANE[] = {0., 0., 1., 1.};
constexpr size_t GRID_SIZE[] = {96, 64};
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

//SparseVector<complex<double>> flattenedRowHamiltionian(const long &i, const long &j) {
//    SparseVector<complex<double>> row(OPERATOR_SIZE);
//    row = VectorXd::Constant(OPERATOR_SIZE, 0.0);
//    row.insert(j * GRID_SIZE[1] + i) = -4;
//    if (i + 1 < GRID_SIZE[0]) {
//        row.insert(j * GRID_SIZE[1] + i + 1) = 1;
//    }
//    if (i - 1 >= 0) {
//        row.insert(j * GRID_SIZE[1] + i - 1) = 1;
//    }
//    if (j + 1 < GRID_SIZE[1]) {
//        row.insert((j + 1) * GRID_SIZE[1] + i) = 1;
//    }
//    if (j - 1 >= 0) {
//        row.insert((j - 1) * GRID_SIZE[1] + i) = 1;
//    }
//    row = row / pow(MESH_STEP, 2);
//    row(j * GRID_SIZE[1] + i) += getPotential(i * MESH_STEP, j * MESH_STEP);
//    return row;
//}

SparseMatrix<complex<double>> getHamiltonian() {
    SparseMatrix<complex<double>> H(OPERATOR_SIZE, OPERATOR_SIZE);
    for (long j = 0; j < GRID_SIZE[1]; ++j) {
        for (long i = 0; i < GRID_SIZE[0]; ++i) {
            //H.block<1, OPERATOR_SIZE>(j * GRID_SIZE[1] + i, 0) = flattenedRowHamiltionian(i, j).transpose();
            H.insert(j * GRID_SIZE[0] + i, j * GRID_SIZE[0] + i) =  -4. / pow(MESH_STEP, 2) + getPotential(i * MESH_STEP, j * MESH_STEP);
            if (i + 1 < GRID_SIZE[0]) {
                H.insert(j * GRID_SIZE[0] + i, j * GRID_SIZE[0] + i + 1) = 1. / pow(MESH_STEP, 2);
            }
            if (i - 1 >= 0) {
                H.insert(j * GRID_SIZE[0] + i, j * GRID_SIZE[0] + i - 1) = 1. / pow(MESH_STEP, 2);
            }
            if (j + 1 < GRID_SIZE[1]) {
                H.insert(j * GRID_SIZE[0] + i, (j + 1) * GRID_SIZE[0] + i) = 1. / pow(MESH_STEP, 2);
            }
            if (j - 1 >= 0) {
                H.insert(j * GRID_SIZE[0] + i, (j - 1) * GRID_SIZE[0] + i) = 1. / pow(MESH_STEP, 2);
            }
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


SparseMatrix<complex<double>> getTildeTSparseMatrix(const size_t &order, const SparseMatrix<complex<double>> &B, vector<SparseMatrix<complex<double>>> &tildeTMatrices) {
//    if (order < tildeTMatrices.size()) {
//        return tildeTMatrices[order];
//    }
    if (order == 0) {
        SparseMatrix<complex<double>> mat(OPERATOR_SIZE, OPERATOR_SIZE);
        mat.setIdentity();
        tildeTMatrices[order] = mat;
        return mat;
    } else if (order == 1) {
        auto mat = B * complex<double>(0, 1);
        tildeTMatrices[order] = mat;
        return mat;
    } else {
        auto mat = B * complex<double>(0, 2) * tildeTMatrices[order - 1] + tildeTMatrices[order - 2];
        tildeTMatrices[order] = mat;
        return mat;
    }
}




SparseMatrix<complex<double>> getEvolutionOperatorForOneTimestep() {
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp declare reduction(matrixAdd : SparseMatrix<complex<double>> : omp_out = omp_out + omp_in)                             \


    const auto H = getHamiltonian();
    //const auto maxEntry = H.cwiseAbs().maxCoeff();
    const auto pointerOfH = H.valuePtr();
    vector<complex<double>> valuesOfH(pointerOfH, pointerOfH + H.nonZeros());
    transform(valuesOfH.begin(), valuesOfH.end(), valuesOfH.begin(), abs<complex<double>>);
    vector<double> normValues(valuesOfH.size());
    for (size_t i = 0; i < valuesOfH.size(); ++i) {
        normValues[i] = valuesOfH[i].real();
    }
    const double maxEntry = *max_element(normValues.begin(), normValues.end());
    double z = TIME_STEP * maxEntry;
    auto B = -H / maxEntry;
    SparseMatrix<complex<double>> evolutionOperator(OPERATOR_SIZE, OPERATOR_SIZE);
    evolutionOperator.setZero();
    complex<double> jv = 1;
    auto properOrder = findProperApproximationOrder(z);
    cout << properOrder << endl;
    vector<SparseMatrix<complex<double>>> tildeTMatrices(properOrder + 1);
    for (size_t i = 0; i <= properOrder; ++i) {
        getTildeTSparseMatrix(i, B, tildeTMatrices);
        cout << i << endl;
    }
    #pragma omp parallel for reduction(matrixAdd : evolutionOperator)
    for (size_t i = 1; i <= properOrder; ++i) {
        cout << i << endl;
        jv = cyl_bessel_j(i, z);
        //Add += when not using parallel
        evolutionOperator = jv * tildeTMatrices[i];
    }
    evolutionOperator *= 2.0;
    evolutionOperator += tildeTMatrices[0] * cyl_bessel_j(0, z);
    return evolutionOperator;
}

//SparseMatrixXcd getEvolutionOperatorForOneTimestep() {
//    auto begin = chrono::system_clock::now();
//    double z = TIME_STEP * maxEntry;
//    auto B = -H / maxEntry;
//    SparseMatrixXcd evolutionOperator;
//    evolutionOperator = SparseMatrixXcd::Zero(OPERATOR_SIZE, OPERATOR_SIZE);
//    complex<double> jv = 1;
//    vector<SparseMatrixXcd> tildeTMatrices = {SparseMatrixXcd::Identity(OPERATOR_SIZE, OPERATOR_SIZE), B * complex<double>(0., 1.)};
//    auto properOrder = findProperApproximationOrder(z);
//    cout << properOrder << endl;
//    for (size_t i = 1; i <= properOrder; ++i) {
//        jv = cyl_bessel_j(i, z);
//        evolutionOperator += jv * tildeTMatrices[0];
//        auto nextTildeT = B * complex<double>(0., 2.) * tildeTMatrices[1] + tildeTMatrices[0];
//        tildeTMatrices[0] = tildeTMatrices[1];
//        tildeTMatrices[1] = nextTildeT;
//        cout << i << endl;
//    }
//    evolutionOperator *= 2.0;
//    evolutionOperator += SparseMatrixXcd::Identity(OPERATOR_SIZE, OPERATOR_SIZE) * cyl_bessel_j(0, z);
//    auto end = chrono::system_clock::now();
//    chrono::duration<double> timeCost = end - begin;
//    cout << "init_minc : " << timeCost.count() << "s." << endl;
//    return evolutionOperator;
//}



const auto evolutionOperator = getEvolutionOperatorForOneTimestep();
VectorXcd currentWave;

VectorXcd propagateWave() {
    currentWave = evolutionOperator * currentWave;
    normalizeWave(currentWave);
    return currentWave;
}

int main() {
    //namespace plt = matplotlibcpp;


    currentWave = getDiscretizedInitialWave();
    for (size_t i = 0; i < 1000; ++i) {
        propagateWave();
        cout << i << endl;
    }
    return 0;
}
