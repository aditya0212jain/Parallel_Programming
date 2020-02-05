#include <malloc.h>
#include <omp.h>
#include <cmath>

float abs(float f){
    if(f>0){
        return f;
    }else{
        return -1*f;
    }
}

struct eigenI{
    float value;
    int index;
};
/*
    * Sorts eigen values and eigenVectors in 
    * decreasing fashion. n is the number of values
*/
void EigenSort(int n,float** eigenValues,float** eigenVectors) 
{ 
   struct eigenI temp[n];
   for(int i=0;i<n;i++){
       temp[i].value = (*eigenValues)[i];
       temp[i].index = i;
    }
    float* sortedEigen = (float*) malloc(sizeof(float)*n);
    float* sortedEigenIndex = (float*) malloc(sizeof(float)*n);
    struct eigenI a;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j){
            if (abs(temp[i].value) < abs(temp[j].value)) 
            {
                a =  temp[i];
                temp[i] = temp[j];
                temp[j] = a;
            }
        }
    }
    for(int i=0;i<n;i++){
        sortedEigen[i] = temp[i].value;
        sortedEigenIndex[i] = temp[i].index;
    }
    *eigenValues = sortedEigen;
    float* sortedEigenVectors = (float*) malloc(sizeof(float)*n*n);
    for(int i=0;i<n;i++){
        int t = sortedEigenIndex[i];
        for(int j=0;j<n;j++){
            sortedEigenVectors[j*n+i] = (*eigenVectors)[j*n+t];
        }
    }
    *eigenVectors = sortedEigenVectors;
} 


/*
    * Prints Matrix D of dim (M,N)
    * where D is an 1D array
*/
void printMat(int M,int N,float* D){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            //printf("%f ",D[i*N+j]);
        }
        //printf("\n");
    }
}

/*
    * Gives the transpose of matrix (M ,N)
    *  output : (N,M) in 1D array
*/
float* getTranspose(int M,int N,float* D){
    float* T;
    T = (float*) malloc(sizeof(float) * N*M);
    #pragma omp parallel for
    for(int i=0;i<M;i++){
        #pragma omp parallel for
        for(int j=0;j<N;j++){
            T[j*M+i] = D[i*N+j];
        }
    }
    return T;
}
/*
    *
    * Returns matrix multiplication of two matrices A and B
    * of dimension (M,N) and (N,P)
    * output : 1D array containing Matrix of dimension
    * (M,P)
    * 
*/
float* getMatMul(int M,int N,int P,float* A,float* B){
    float* C;
    C = (float*) malloc(sizeof(float)* M*P);
    // int i=0,j=0;
    #pragma omp parallel for
    for(int i=0;i<M;i++){
        for(int j=0;j<P;j++){
            float sum=0;
            for(int k=0;k<N;k++){
                sum += A[i*N+k]*B[k*P+j];
            }
            C[i*P+j] = sum;
        }
    }
    return C;
}

float* getEye(int M,int N){
    float* E = (float*) malloc(sizeof(float)* M*N);
    #pragma omp for
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            E[i*N+j] = 0;
        }
    }
    for(int i=0;i<M;i++){
        E[i*N+i] = 1;
    }
    return E;
}

void getGivensRotation(float a,float b,float* c,float* s){
    if(b==0){
        *c = 1;
        *s = 0;
    }else{
        *c = a/sqrt(a*a + b*b);
        *s = (-1*b)/sqrt(a*a + b*b);
    }
    return;
}

void getQRfactorsGivens(int M,int N,float* A,float** Q,float** R){
    // free(*R);
    float* Qtemp = getEye(M,M);
    float* Rtemp = (float*) malloc(sizeof(float)* M*N);
    float* v1 = (float*) malloc(sizeof(float)*N);
    float* v2 = (float*) malloc(sizeof(float)*N);
    float* w1 = (float*) malloc(sizeof(float)*N);
    float* w2 = (float*) malloc(sizeof(float)*N);
    free(*Q);
    free(*R);

    #pragma omp parallel for collapse(2)
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            Rtemp[i*N+j] = A[i*N+j];
        }
    }

    int pomp = 1;
    for(int j=0;j<N;j++){
        for(int i=M-1;i>j;i--){
            float cosine,sine;
            getGivensRotation(Rtemp[j*N+j],Rtemp[i*N+j],&cosine,&sine);
                for(int k=0;k<N;k++){
                    v1[k] = cosine*Rtemp[j*N+k] + (-1*sine)*Rtemp[i*N+k];
                    v2[k] = sine*Rtemp[j*N+k] + cosine*Rtemp[i*N+k];
                    
                    w1[k] = cosine*Qtemp[j*N+k] + (-1*sine)*Qtemp[i*N+k];
                    w2[k] = sine*Qtemp[j*N+k] + cosine*Qtemp[i*N+k];
                }

                for(int k=0;k<N;k++){
                    Rtemp[j*N+k] = v1[k];
                    Rtemp[i*N+k] = v2[k];
                    Qtemp[j*N+k] = w1[k];
                    Qtemp[i*N+k] = w2[k];
                }
        }
    }
    *Q = getTranspose(M,M,Qtemp);
    *R = Rtemp;
}

/*
    *
    * Takes matrics of dim (M,N) and computes 
    * eigenValues (N) and eigenVectors (N,N)
    * 
*/
void QRAlgorithm(int M,int N,float* D,float** eigenValues,float** eigenVectors){
    float* D0 = (float*) malloc(sizeof(float)* N*N);
    float* E0 = (float*) malloc(sizeof(float)* N*N);
    float* D0old = (float*) malloc(sizeof(float)* N*N);
    float* E0old = (float*) malloc(sizeof(float)*N*N);
    float* SigmaTemp = getEye(M,M);

    #pragma omp parallel for collapse(2) 
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            D0[i*N+j] = D[i*N+j];
            E0[i*N+j] = 0;
            E0old[i*N+j] = 0;
        }
    }

    #pragma omp parallel for
    for(int i=0;i<N;i++){
        E0[i*N+i] = 1;
        E0old[i*N+i] = 1;
    }
    bool converged = false;
    float* Q = (float*)malloc(sizeof(float)* N);
    float* R = (float*)malloc(sizeof(float)* N);
    int limit = 4000;
    int it = 0;
    int flag = false;
    float diff = 0;
    float temp1=0;
    float diff2old=0;
    while(!converged && it<limit){
        getQRfactorsGivens(M,N,D0,&Q,&R);
        
        // //printf("m");
        D0 = getMatMul(N,N,M,R,Q);
        E0 = getMatMul(N,N,M,E0,Q);
        // //printf("n\n");
        diff = 0;
        converged = true;

        ////Checking convergence
        for(int i =0;i<N;i++){
            for(int j=0;j<N;j++){
                temp1 = abs(E0old[i*N+j] - E0[i*N+j]);
                diff+=temp1;
                if(temp1>0.001){
                    converged = false;
                    // break;
                }
            }
        }
        //printf("diff : %f\n",diff);
        
        // #pragma omp parallel for collapse(2)
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                // D0old[i*N+j] = D0[i*N+j];
                E0old[i*N+j] = E0[i*N+j];
            }
        }
        it++;
    }
    float* eigenValuesTemp = (float*) malloc(sizeof(float)*N);
    for(int i=0;i<N;i++){
        eigenValuesTemp[i] = D0[i*N+i];
    }
    *eigenValues = eigenValuesTemp;
    *eigenVectors = E0;
}

void SVD_Helper(int M, int N, float* D, float** U, float** SIGMA, float** V_T, float* DT)
{
    float* D_T;
    float* D_DT;
    float* eigenValues;
    float* eigenVectors;
    float* SIGMAtemp = (float*) malloc(sizeof(float)*N*M);//(M,N)
    float* SIGMA_inv = (float*) malloc(sizeof(float)*N*M);//(N,M)
    float* Utemp = (float*) malloc(sizeof(float)*M*M);
    float* Vtemp = (float*) malloc(sizeof(float)*N*N);
    float* Vtemp1  = (float*) malloc(sizeof(float)*N*M);
    float* ourD;
    // float* Utemp1 = (float*) malloc(sizeof(float)*M*N);
    ////////////////////////////////////////////////////
    //printf("M: %d,N: %d \n",M,N);
    //////////////////////////////////////////////////////
    // M << N so compute D_DT then the values of the eigenvectors will be U and then 
    // V_T = SigmInv*U^T*D (N,M)*(M,M)*(M,N) = (N,N)
    // D_T = getTranspose(M,N,D);
    D_T = DT;
    D_DT = getMatMul(M,N,M,D,D_T);
    
    //DT_D is of the dimension (N,N)
    QRAlgorithm(M,M,D_DT,&eigenValues,&eigenVectors);
    EigenSort(M,&eigenValues,&eigenVectors);

    *SIGMA = SIGMAtemp;
    *U = eigenVectors;
    Utemp = eigenVectors;
    
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                SIGMAtemp[i*N+j] = 0;
                SIGMA_inv[j*M+i] = 0;
            }
        }
        
        #pragma omp for
        for(int i=0;i<M;i++){
            SIGMAtemp[i*N+i] = sqrt(eigenValues[i]);
        }

        #pragma omp for
        for(int i=0;i<M;i++){
            SIGMA_inv[i*M+i] = 1/SIGMAtemp[i*N+i];
        }

        #pragma omp for collapse(2)
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                Vtemp[i*N+j] = 0;
            }
        }

        #pragma omp for
        for(int i=0;i<M;i++){
            for(int j=0;j<M;j++){
                SIGMA_inv[i*M+j] = (1/SIGMAtemp[i*N+i])*Utemp[j*M+i];
            }
        }

        #pragma omp for 
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                float su1 = 0;
                for(int k=0;k<M;k++){
                    su1 += SIGMA_inv[i*M+k]*D[k*N+j];
                }
                Vtemp[i*N+j] = su1;
            }
        }
    }

    // V_T = SigmInv*U^T*D (N,M)*(M,M)*(M,N) = (N,N)
    *V_T = Vtemp;

}



// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
    omp_set_num_threads(12);
    bool svd_of_transpose = true;
    float* Dtemp;
    if(svd_of_transpose){
        // //printf("Converting to D^T\n");
        Dtemp = getTranspose(M,N,D);
    }
    SVD_Helper(N,M,Dtemp,U,SIGMA,V_T,D);
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    float sum = 0;
    float total_sum = 0;
    int k=1;
    

    #pragma parallel omp for reduction(+:total_sum)
    for(int i=0;i<N;i++){
        total_sum += SIGMA[i*M+i];
    }

    for(int i=0;i<N;i++){
        sum+= SIGMA[i*M+i]/total_sum;
        if((100*sum)>=retention){
            break;
        }
        k++;
    }
    
    float* projectionMatrix = (float*) malloc(sizeof(float)*k*N);

    #pragma omp parallel for    
    for(int i=0;i<k;i++){
        for(int j=0;j<N;j++){
            projectionMatrix[j*k+i] = U[j*N+i];
        }
    }
    
    float* Dtemp = getMatMul(M,N,k,D,projectionMatrix);// (M,k) = (M,N) * (N,k)
    *D_HAT = Dtemp;
    *K = k;
}
