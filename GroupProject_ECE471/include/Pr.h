/*
 * pr.h - header file of the pattern recognition library
 *
 * Author: Hairong Qi, ECE, University of Tennessee
 *
 * Date: 01/25/04
 *
 * Please send all your comments to hqi@utk.edu 
 * 
 * Modified:
 *   - 09/24/13: add "const" to the filename parameters to remove warning 
 *               msg in new compilers (Steven Clukey)
 *   - 04/26/05: reorganized for the Spring 2005 classs
 *   - Niloo Ranjan: 10/07/2015: added the header for additional functions used for project 2
 *   - Niloo Ranjan: 10/28/2015: added the header for additional functions used for project 3
 *
 */

#ifndef _PR_H_
#define _PR_H_

#include "Matrix.h"


/////////////////////////  
// file I/O
Matrix readData(const char *,            // the file name
                int);                    // the number of columns of the matrix
Matrix readData(const char *,            // the file name
                int,                     // the number of columns
                int);                    // the number of rows (or samples)
Matrix readData(const char *);           // read data file to a matrix with 1 row
void writeData(Matrix &, const char *);  // write data to a file
Matrix readImage(const char *,           // read the image from a file
                 int *,                  // the number of rows (or samples)
                 int *);                 // the number of columns
void writeImage(const char *,            // write the image to a file
                Matrix &,                // the matrix to write
                int,                     // the number of rows
                int);                    // the number of columns


////////////////////////
// distance calculation
double euc(const Matrix &,         // Euclidean distance between two vectors
	   const Matrix &);
double mah(const Matrix &,         // the Mahalanobis distance, input col vec
	   const Matrix &C,        // the covariance matrix
	   const Matrix &mu);      // the mean (a col vector)


////////////////////////
// classifiers

// maximum a-posteriori probability (mpp)
int mpp(const Matrix &train,        // the training set of dimension mx(n+1)
                                    // where the last col is the class label
                                    // that starts at 0
        const Matrix &test,         // one test sample (a col vec), nx1
        const int,                  // number of different classes
	const int,                  // caseI,II,III of the discriminant func
	const Matrix &Pw);          // the prior prob, a col vec

// this function returns the optimal projection direction W for FLD
Matrix GetFLD ( Matrix &train);     

// this function returns the optimal basis vector for PCA 
Matrix GetPCA( Matrix &nX);

// this function transformed the normalized training and the testing data for FLD and PCA
Matrix GetDataTransformedFLD ( Matrix &nX, Matrix W);

// this function calculates the TP, TN, FP, FN, TPR, FPR, sensitivity, specificity, precision, accuracy, and recall rate 
void DerivePerferformanceMetric ( Matrix & tested, Matrix &truth, int datatype);

// this function does the classification on data set nX, tX, fX using MAP with 
// optimal prior probability found 
void ClassificationWithBestPW ( Matrix & trn, Matrix &tet, Matrix &BestPW, int type );

// this function implements the basic implementattion of the kNN using 
// original Euclidean distance to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively
int KNNClassifierEuclidian(Matrix &nXTr, Matrix &sample, int K);

// this method gets the unsorted computed K distance 
// and sorts the K distances using insersort() method from 
// provided matrix library
Matrix ComputeSortedDis( Matrix &UnSortDis);

// this function calculates the partial Euclidean distance to be used for 
// kNN implementation using Partial Euclidian distance
double calculatePartDis(Matrix &s1, Matrix &s2);

// this function implements the kNN using partial Euclidean distance 
// to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively
int KNNClassifierPartialEuclidian(Matrix &nXTr, Matrix &sample, int K);

// this function implements the kNN using original Minkowski distance 
// to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively
int KNNClassifierMinkowski(Matrix &nXTr, Matrix &sample, int K, double MinK);

// this function implements the kNN using partial Minkowski distance 
// to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively
int KNNClassifierPartialMinkowski(Matrix &nXTr, Matrix &sample, int K, double minK);

// this function calculates the partial Minkowski distance to be used for 
// kNN implementation using Partial Minkowski distance
double calculatePartDisMinkowski(Matrix &s1, Matrix &s2, double minK);

// this method build the validating data set for 10-fold cross-validation
// with kNN as classifier
Matrix getTestingData(Matrix &S, Matrix &glassData);

// this method build the training data set for 10-fold cross-validation
// with kNN as classifier
Matrix getTrainingData(Matrix &foldData, Matrix &glassData, int i );

// this function implements kNN using original Euclidean distance to be used to 
// clasiify the testing set using 10 folds cross-validation technique
int KNNClassifierEuclidianFold(Matrix &nXTr, Matrix &sample, int K,int classes);

#endif

