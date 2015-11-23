/********************************************************************
 ** mpp.cpp
 **
 ** Purpose: Supervised learning: 
 **    Maximum posterior probability (MPP). This algorithm assumes 
 **    Gaussian distribution and zero-one loss 
 ** 
 ** Prototype: int mpp(Matrix train, Matrix test, 
 **                    int class, int case, Matrix Pw) 
 **    - train: the training set of m x (n+1) matix
 **             where m is the nr of rows (or samples)
 **             n is the number of features
 **             the last column is the class label, starting at 1
 **    - test: the testing sample to be classified, a column vector
 **            with a dimension nx1
 **    - class: number of different classes, assuming the class labels
 **             starts at 0
 **    - case: 1, 2, 3, 4 - case I, II or III and likelihood ratio test
 **      cases==1: The features are statistically 
 **                indepdent, and have the same variance.
 **                Can be simplified to minimum distance.
 **      cases==2: The covariance matrices for all classes are
 **                identical, but not a scalar of identity matrix.
 **                We pick the average.
 **      cases==3: The most general case. The covariance matrices
 **                are not equal from each other.
 **      cases==4: The likelihood ratio to derive decision rule
 **    - Pw: Prior probability. A column vector of dimension cx1
 ** 
 **    - output: the class label of the input test sample
 **
 ** Created by: Hairong Qi (hqi@utk.edu)
 **
 ** Modified by
 **   - 02/20/2008: the class label starts at 0 instead of 1
 ** Modified by
 **   - Niloo Ranjan
 **   - 09/10/2015: added fourth case for deriving decision rule using 
 **                  likelihood ratio
 **
 ************************************************************************/

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "Pr.h"


using namespace std;

const double PI = 3.14159265;

int mpp(const Matrix &train, const Matrix &tedata, const int c, const int cases, const Matrix &Pw)
{
    static int first = 1, nf;
    static double varavg;
    static Matrix *means, *covs, covavg;
    int nctr, ncte, nrtr, nrte;
    int i, j;
    Matrix covsum, tmp;
    double sum;

    //////////////////////////////////////////////////////////////////////
    // calculate the means and covs only when the function is called the 1st time
    if (first==1) {
        // get the size of input raw data
        nctr = train.getCol();
        ncte = tedata.getCol();
        nrtr = train.getRow();
        nrte = tedata.getRow();
        if (nctr != (nrte+1)) {
            cout << "MPP: "
                << "Training and testing set do not have same number of features\n";
            exit(3);
        }
        nf = nctr-1;

        // calculate the mean and covariance of each class
        // the mean is a cxnf matrix and the cov is a c*nf x nf matrix
        means = (Matrix *) new Matrix [c];
        for (i=0; i<c; i++)
            means[i].createMatrix(nf, 1);
        covs = (Matrix *) new Matrix [c];
        for (i=0; i<c; i++)
            covs[i].createMatrix(nf, nf);

        // the following two matrices are used for case 2
        // the average covariance matrix is used as the common matrix
        covsum.createMatrix(nf,nf);
        covavg.createMatrix(nf,nf);

        for (i=0; i<c; i++) {
            tmp = getType(train, i);
            covs[i] = cov(tmp, nf);
            means[i] = mean(tmp, nf);
            covsum = covsum + covs[i];
        }

        // calculate the average covariance to be used by case II
        covavg = covsum / (double)c;

        // calculate the average variance to be used by case I
        sum = 0.0;
        for (i=0; i<c; i++)
            sum += covavg(i,i);
        varavg = sum / (double)c;

        first++;
    }

    //////////////////////////////////////////////////////////
    // classification
    Matrix disc(1,c), sdisc(1,c), pos(1,c);
    double mdist, edist;

    // find the discriminant function value
    switch (cases) {
        case 1:
            for (i=0; i<c; i++) {        // for each class
                edist = euc(tedata, means[i]);
                disc(0,i) = -edist*edist/(2*varavg) + log(Pw(i,0));
            }
            break;
        case 2:
            for (i=0; i<c; i++) {
  //          cout << "Means here:  " << means[i] << "covar:   " << covs[i] << endl;
                mdist = mah(tedata, covavg, means[i]);
                disc(0,i) = -0.5*mdist*mdist + log(Pw(i,0));
            }
            break;
        case 3:
            for (i=0; i<c; i++) {
    //        cout << "Means here:  " << means[i] << "covar:   " << covs[i] << endl;
                mdist = mah(tedata, covs[i], means[i]);
                disc(0,i) = -0.5*mdist*mdist - 0.5*log(det(covs[i])) + log(Pw(i,0));
            }
            break;
        case 4:
            for (i=0; i<c; i++) {
                Matrix diff = (tedata - (means[i]));
                Matrix diffTrans = transpose(diff);
                Matrix inverseCovar = inverse(covs[i]);
                double detCovar = det(covs[i]);
            cout << "invercovariance here:  " << inverseCovar << "covar:   " << covs[i] << "powdethalf: " << pow(detCovar, 0.5) << endl;
                
                Matrix matrixMul = (diffTrans ->* inverseCovar) ->* diff;
                //double ret = mtod((diffTrans ->* inverseCovar) ->* diff);
                //Matrix result = matrixMul * (-0.5);
                double ret = mtod(matrixMul * (-0.5));
                //disc(0,i) = (1.0/(2*PI*pow(detCovar, 0.5)))*(exp(result(0,0)));
                disc(0,i) = (1.0/(2*PI*pow(detCovar, 0.5)))*(exp(ret));
            }
            break;
    }
    // sort the discriminant function value in the ascending order
    insertsort(disc, sdisc, pos);
    // return the label of the class with the largest discriminant value
    return (int)pos(0,c-1);
}
