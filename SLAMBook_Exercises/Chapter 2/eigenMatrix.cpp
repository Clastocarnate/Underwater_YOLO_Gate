#include <iostream>
using namespace std;
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#define MATRIX_SIZE 50
//argc = Argument count, argv = Argument Vector
//Used for giving Arguments in the command line
int main(int argc, char** argv) {
    // Eigen::Matrix is a template class that defines a matrix.
    // Here, matrix_23 is declared as a 2x3 matrix of floats.
    Eigen::Matrix<float, 2, 3> matrix_23;

    // Eigen::Vector3d is a typedef for Eigen::Matrix<double, 3, 1>
    // It defines a 3-dimensional vector with double precision.
    Eigen::Vector3d v_3d;

    // Eigen::Matrix<float, 3, 1> is a 3x1 matrix (or vector) of floats.
    // vd_3d is a 3-dimensional vector of floats.
    Eigen::Matrix<float, 3, 1> vd_3d;

    // Eigen::Matrix3d is a typedef for Eigen::Matrix<double, 3, 3>
    // It defines a 3x3 matrix of doubles.
    // Here, matrix_33 is initialized to be a 3x3 matrix of zeros.
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();

    // This declares a dynamic-size matrix of doubles.
    // Eigen::Dynamic allows the matrix size to be determined at runtime.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;

    // Eigen::MatrixXd is a typedef for a dynamic-size matrix of doubles.
    // It is a shorthand for Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>.
    Eigen::MatrixXd matrix_x;

    // Initializing the matrix_23 with values 1, 2, 3 in the first row,
    // and 4, 5, 6 in the second row.
    matrix_23 << 1, 2, 3, 4, 5, 6;

    cout << matrix_23 << endl;

    // Initialize matrix_33 with random values.
    // Eigen::Matrix3d::Random() generates a 3x3 matrix of random values.
    matrix_33 = Eigen::Matrix3d::Random();
    cout << matrix_33 << endl << endl;
    cout << "Transpose" << endl;
    cout << matrix_33.transpose() << endl;
    cout << "Sum" << endl;
    cout <<matrix_33.sum() << endl;
    cout << "Trace" <<endl;
    cout << matrix_33.trace() << endl;
    cout << "Scalar Multiplication"<< endl;
    cout << 10*matrix_33 << endl;
    cout << "Inverse" << endl;
    cout << matrix_33.inverse() << endl;
    cout << "Determinant" << endl;
    cout << matrix_33.determinant() << endl;
    // Eigen::SelfAdjointEigenSolver is a class in the Eigen library used to compute 
    // eigenvalues and eigenvectors of self-adjoint (i.e., symmetric or Hermitian) matrices.
    // Here, it is instantiated with Eigen::Matrix3d, which is a 3x3 matrix of doubles.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);

    // The eigen_solver object now contains the eigenvalues and eigenvectors of the matrix 
    // formed by the product of the transpose of matrix_33 and matrix_33 itself.

    // Output the computed eigenvalues to the console.
    // The eigenvalues are the roots of the characteristic polynomial of the matrix, 
    // and they indicate important properties of the matrix, such as its scaling along certain directions.
    cout << "Eigen Values: " << eigen_solver.eigenvalues() << endl;

    // Output the computed eigenvectors to the console.
    // The eigenvectors correspond to the directions in which the matrix acts as a simple scaling factor.
    // Each eigenvector is associated with one of the eigenvalues.
    cout << "Eigen Vectors: " << eigen_solver.eigenvectors() << endl;

    // The program returns 0 indicating successful execution.
    return 0;
}