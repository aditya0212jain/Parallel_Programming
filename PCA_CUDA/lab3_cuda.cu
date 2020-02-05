#include "lab3_cuda.h"
#include <omp.h>
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_SIZE 32

/*
    * Prints Matrix D of dim (M,N)
    * where D is an 1D array
*/
void printMat(int M,int N,double* D){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            printf("%f ",D[i*N+j]);
        }
        printf("\n");
    }
}


double L2(int M, int N, double* D){
	double sum=0.0;
	for(int j=0;j<N*M;j++)
		sum+= *(D+j) * *(D+j);

	return sqrt(sum);
}


struct eigenI{
    double value;
    int index;
};

int myCeil(int a, int b){
    if(a%b==0){
        return a/b;
    }else{
        return (a/b)+1;
    }
}

/*
    * Gives the transpose of matrix (M ,N)
    *  output : (N,M) in 1D array
*/
double* getTranspose(int M,int N,double* D){
    double* T;
    T = (double*) malloc(sizeof(double) * N*M);
    // #pragma omp parallel for
    for(int i=0;i<M;i++){
        // #pragma omp parallel for
        for(int j=0;j<N;j++){
            T[j*M+i] = D[i*N+j];
        }
    }
    return T;
}

void getEye(double* E,int N){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            E[i*N+j] = 0;
        }
    }
    for(int i = 0 ;i<N;i++){
        E[i*N+i]= 1;
    }
}

/*
    *
    * Returns matrix multiplication of two matrices A and B
    * of dimension (M,N) and (N,P)
    * output : 1D array containing Matrix of dimension
    * (M,P)
    * 
*/
double* getMatMul(int M,int N,int P,double* A,double* B){
    double* C;
    C = (double*) malloc(sizeof(double)* M*P);
    // int i=0,j=0;
    // #pragma omp parallel for
    for(int i=0;i<M;i++){
        for(int j=0;j<P;j++){
            double sum=0;
            for(int k=0;k<N;k++){
                sum += A[i*N+k]*B[k*P+j];
            }
            C[i*P+j] = sum;
        }
    }
    return C;
}

bool isEqualDouble(double a, double b){
    if(abs(a-b)<0.001){
        return true;
    }
    return false;
}

/*
    * Sorts eigen values and eigenVectors in 
    * decreasing fashion. n is the number of values
*/
void EigenSort(int n,double** eigenValues,double** eigenVectors) 
{ 
   struct eigenI temp[n];
   for(int i=0;i<n;i++){
       temp[i].value = (*eigenValues)[i];
       temp[i].index = i;
    }
    double* sortedEigen = (double*) malloc(sizeof(double)*n);
    double* sortedEigenIndex = (double*) malloc(sizeof(double)*n);
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
    double* sortedEigenVectors = (double*) malloc(sizeof(double)*n*n);
    for(int i=0;i<n;i++){
        int t = sortedEigenIndex[i];
        for(int j=0;j<n;j++){
            sortedEigenVectors[j*n+i] = (*eigenVectors)[j*n+t];
        }
    }
    *eigenVectors = sortedEigenVectors;
} 

/*
    * Jacobian functions 
*/
int maxind(int k,int N,double* S){
    int m = k+1;
    for(int i=k+2;i<N;i++){
        if(abs(S[k*N+i])>abs(S[k*N+m])){
            m = i;
        }
    }
    return m;
}

void update(int k,double t,bool* changed,int* state,double* e){
    double prev_ek = e[k];
    e[k] = e[k] + t;
    if(changed[k] && isEqualDouble(prev_ek,e[k])){
        changed[k] = false;
        *state = *state -1;
    }
    else if(!changed[k] && !isEqualDouble(prev_ek,e[k])){
        changed[k] = true;
        *state = *state + 1;
    }
}

void rotate(int k,int l,int i,int j,double c,double s,double* S,int JDim){
    double temp1 = S[k*JDim + l];
    double temp2 = S[i*JDim + j];
    S[k*JDim+l] = c*temp1 - s*temp2;
    S[i*JDim+j] = s*temp1 + c*temp2;
}

void rotateE(int i,int k,int l,double c,double s,double* E,int JDim){
    double temp1 = E[i*JDim+k];
    double temp2 = E[i*JDim+l];
    E[i*JDim+k] = c*temp1 - s*temp2;
    E[i*JDim+l] = s*temp1 + c*temp2;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                        CUDA FUNCTIONS 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//d_M (a,b)
//d_N (b,c)
//d_P (a,c)
__global__ void simpleMatMulKernell(double* d_M, double* d_N, double* d_P, int a,int b,int c){
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    // printf("row : %d , col : %d \n",row,col);
    // printf("blockIDx.x : %d blockDim.x : %d threadIdx.x : %d blockIdx.y : %d blockDim.y %d threadIdx.y : %d\n",blockIdx.x,blockDim.x,threadIdx.x,blockIdx.y,blockDim.y,threadIdx.y);
    if(row<a && col<c) {
        double product_val = 0;
        for(int k=0;k<b;k++) {
           product_val += d_M[row*b+k]*d_N[k*c+col];
        }
        d_P[row*c+col] = product_val;
    }
}

__global__ void efficientMatMulKernell(double* d_M, double* d_N, double* d_P, int a,int b,int c){
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  double  Pvalue = 0;

  int row = threadIdx.y;
  int col = threadIdx.x;

  int element_col = blockIdx.x*blockDim.x+threadIdx.x;
  int element_row = blockIdx.y*blockDim.y+threadIdx.y;
  
  int rolls = b%BLOCK_SIZE==0 ? b/BLOCK_SIZE : (b/BLOCK_SIZE) + 1 ;

  __shared__ double Ms[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double Ns[BLOCK_SIZE][BLOCK_SIZE];

  // printf("rolls: %d\n",rolls);
  for(int t = 0;t<rolls;t++){
    int Mrow = blockRow*BLOCK_SIZE;
    int Mcol = t*BLOCK_SIZE ;
    int Nrow = Mcol;
    int Ncol = blockCol*BLOCK_SIZE;
     
    int M_offset = Mrow*b + Mcol;// check for b
    int N_offset = Nrow*c + Ncol;

    if((Mrow + row)<a && (Mcol + col)<b){
      Ms[row][col] = d_M[M_offset+ row*b + col];
    }else{
      Ms[row][col] = 0;
    }
    if((Nrow + row)<b && (Ncol + col)<c){
      Ns[row][col] = d_N[N_offset+ row*c + col];
    }else{
      Ns[row][col] = 0;
    }
    __syncthreads();
    
    for(int l = 0;l<BLOCK_SIZE;l++){
	Pvalue += Ms[row][l]*Ns[l][col];
    }
    
    __syncthreads();
    
  }

  if(element_row < a && element_col < c){
    d_P[element_row*c + element_col] = Pvalue;
  }
  
}

double* launchMatMulCUDA(int A,int B,int C,double* X,double* Y){
    // X (A,B)
    // Y (B,C)
    // result (A,C)
    double *d_Xdata, *d_Ydata, *d_Zdata;
    double *h_Zdata = (double *) malloc(sizeof(double)*A*C);
    int a = myCeil( A , BLOCK_SIZE );
    int c = myCeil( C , BLOCK_SIZE );
    // printf(" a: %d , c: %d\n",a,c);
    dim3 dimGrid( c , a , 1);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE, 1);
    /////////////////////////////////////////////////////////////////
    cudaMalloc(&d_Xdata,sizeof(double)*A*B);
    cudaMalloc(&d_Ydata,sizeof(double)*B*C);
    cudaMalloc(&d_Zdata,sizeof(double)*A*C);

    cudaMemcpy(d_Xdata, X, sizeof(double)*A*B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ydata,Y,sizeof(double)*B*C,cudaMemcpyHostToDevice);

    efficientMatMulKernell<<<dimGrid,dimBlock>>> (d_Xdata,d_Ydata,d_Zdata,A,B,C);
    // simpleMatMulKernell<<<dimGrid,dimBlock>>> (d_Xdata,d_Ydata,d_Zdata,A,B,C);
    cudaMemcpy(h_Zdata,d_Zdata,sizeof(double)*A*C,cudaMemcpyDeviceToHost);
    cudaFree(d_Zdata);
    cudaFree(d_Ydata);
    cudaFree(d_Xdata);
    return h_Zdata;
}

// d_In (m,n)
// d_Out (n,m)
__global__ void simpleTranspose(double* d_In,double* d_Out,int m,int n){
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    
    if(row<m&&col<n){
        d_Out[col*m+row] = d_In[row*n + col];
    }
}


// d_In (m,n)
// d_Out (n,m)
__global__ void efficientTranspose(double* d_In,double* d_Out,int m,int n){
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS){
    if((y+j)<m && x<n){
      d_Out[x*width + (y+j)] = d_In[(y+j)*width + x];
    }
  }
}


double* launchTransposeCUDA(int m,int n,double* D){
    double *d_In, *d_Out;
    int memsize = sizeof(double)*m*n;
    double *h_out = (double *)malloc(memsize);
    dim3 dimGrid(myCeil(m,32),myCeil(n,32), 1);
    dim3 dimBlock(32,32, 1);
    ////////////////////////////////////////////////////////////////////////
    cudaMalloc(&d_In,memsize);
    cudaMalloc(&d_Out,memsize);
    ////////////////////////////////////////////////////////////////////////
    cudaMemcpy(d_In,D,memsize,cudaMemcpyHostToDevice);
    ///////////////////////////////////////////////////////////////////////
    simpleTranspose<<<dimGrid,dimBlock>>>(d_In,d_Out,m,n);
    ////////////////////////////////////////////////////////////////////////
    cudaMemcpy(h_out,d_Out,memsize,cudaMemcpyDeviceToHost);
    //////////////////////////////////////////////////////////////////////
    cudaFree(d_In);
    cudaFree(d_Out);
    return h_out;
}

__global__ void rotateCUDA(int k,int l,double c,double s,double* d_S,double* d_E,int JDim){
    int i =blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=0 && i < JDim ){
        double temp1 = d_E[i*JDim+k];
        double temp2 = d_E[i*JDim+l];
        d_E[i*JDim+k] = c*temp1 - s*temp2;
        d_E[i*JDim+l] = s*temp1 + c*temp2;
    }
        
    int a,b,x,y;
    if(i>=0 && i < k){
        a = i;
        b = k;
        x = i;
        y = l;
    }else if(i>=k+1 && i< l){
        a = k;
        b = i;
        x = i;
        y = l;
    }else if(i>=l+1 && i < JDim){
        a = k;
        b = i;
        x = l;
        y = i;
    }

    if(i>=0 && i<JDim){
        double temp1 = d_S[a*JDim + b];
        double temp2 = d_S[x*JDim + y];
        d_S[a*JDim+b] = c*temp1 - s*temp2;
        d_S[x*JDim+y] = s*temp1 + c*temp2;
    }

}

void launchRotateCUDA(int k,int l,double c,double s,double* S,double* E,int JDim){
    double *d_S, *d_E;
    int memsize = sizeof(double)*JDim*JDim;
    dim3 dimGrid(myCeil(JDim,32),1, 1);
    dim3 dimBlock(32,1,1);

    cudaMalloc(&d_S,memsize);
    cudaMalloc(&d_E,memsize);
    cudaMemcpy(d_S,S,memsize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_E,E,memsize,cudaMemcpyHostToDevice);

    rotateCUDA<<<dimGrid,dimBlock>>>(k,l,c,s,d_S,d_E,JDim);

    cudaMemcpy(S,d_S,memsize,cudaMemcpyDeviceToHost);
    cudaMemcpy(E,d_E,memsize,cudaMemcpyDeviceToHost);
    cudaFree(d_S);
    cudaFree(d_E);
}

/*
    * The Jacobi algorithm for finding eigenValues and eigenVectors
    * OUT : e , E 
    * IN  : N , S
*/
void jacobi(int N,double* S,double* e,double* E){
    /*
        * Global Variables for Jacobi
    */
    int state;
    int JDim = N;//dimension of the square matrix whose jacobian is to be calculated
    bool* changed = (bool*) malloc(sizeof(bool)* JDim);
    int* ind = (int*) malloc(sizeof(int)* JDim);

    state = N;
    getEye(E,JDim);

    //Initializaiton of jacobi
    for(int k=0;k<N;k++){
        ind[k] = maxind(k,JDim,S);
        e[k] = S[k*JDim + k];
        changed[k] = true;
    }

    int m=0;
    int iter=0;
    int x,k,l;
    int i;
    double p,y,d,r,c,s,t,p_square;
    while(state!= 0){
        m=0;
        // printf("iter: %d \n",iter);
        iter++;
        for(x=1;x<JDim;x++){
            if(abs(S[x*JDim+ind[x]])>abs(S[m*JDim+ind[m]])){
                m = x;
            }
        }
        k = m;
        l = ind[m];
        p = S[k*JDim+l];
        y = (e[l] - e[k])/2;
        p_square = p*p;
        d = abs(y) + sqrt(p_square + y*y);
        r = sqrt(p_square + d*d);
        c = d/r;
        s = p/r;
        t = p_square/d;
        if(y<0){
            s *= -1;
            t *= -1;
        }
        S[k*JDim + l ] = 0.0;
        update(k,-1*t,changed,&state,e);
        update(l,t,changed,&state,e);

        
        for(i=0;i<k;i++){
            rotate(i,k,i,l,c,s,S,JDim);
        }
        
        for(i=k+1;i<l;i++){
            rotate(k,i,i,l,c,s,S,JDim);
        }
        
        for(i=l+1;i<JDim;i++){
            rotate(k,i,l,i,c,s,S,JDim);
        }

        for(i=0;i<N;i++){
            rotateE(i,k,l,c,s,E,JDim);
        }

        // launchRotateCUDA(k,l,c,s,S,E,JDim);

        ind[k] = maxind(k,JDim,S);
        ind[l] = maxind(l,JDim,S);
    }
    
}

void check_matmul_correctness(double* Dc,int A,int B,int C,double* M,double* N){
  // printf("checking matmul between simple and efficient \n");
  double* test1 = getMatMul(A,B,C,M,N);
  int count =0;
  for(int i=0;i<A;i++){
    for(int j=0;j<C;j++){
      if(isEqualDouble(Dc[i*C+j],test1[i*C+j])==false){
	count++;
	// printf("%lf %lf %lf\n",Dc[i*C+j],test1[i*C+j],Dc[i*C+j] - test1[i*C+j]);
      }
    }
  }
  printf("total incorrect : %d\n",count);
}

void check_diff(int A,int B,double* M,double* N){
  int count=0;
  for(int i=0;i<A;i++){
    for(int j=0;j<B;j++){
      if(isEqualDouble(M[i*B+j],N[i*B+j])==false){
	count++;
	printf("%lf %lf \n",M[i*B+j],N[i*B+j]);
      }
    }
  }
  // printf("checked for diff between simple and efficient with : %d \n",count);
}

double* get_Dhat(double* D,int M,int N,int K,double * ProjectionM){
  return launchMatMulCUDA(M,N,K,D,ProjectionM);
}

void SVD_and_PCA (int M, 
        int N, 
        double* D, 
        double** U, 
        double** SIGMA, 
        double** V_T, 
        int* SIGMAm,
        int* SIGMAn, 
        double** D_HAT, 
        int *K,
        int retention) {
    // write your code here
    double* D_T = launchTransposeCUDA(M,N,D);
    double* DT_D = launchMatMulCUDA(N,M,N,D_T,D);
    
    double* e = (double*) malloc(sizeof(double)* N);
    double* E = (double*) malloc(sizeof(double)* N*N);

    jacobi(N,DT_D,e,E);

    EigenSort(N,&e,&E);
    double* Sigma = (double*) malloc(sizeof(double)* N*M);
    double* SigmaInverse = (double*) malloc(sizeof(double)* N*M);
    double* sigma_new = (double*) malloc(sizeof(double)*N);

    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            Sigma[i*M+j] = 0;
        }
        Sigma[i*M+i] = sqrt(abs(e[i]));
	sigma_new[i] = Sigma[i*M+i];
    }
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            SigmaInverse[i*N+j] = 0;
        }
    }
    for(int i=0;i<N;i++){
        SigmaInverse[i*N+i] = 1/Sigma[i*M+i];
    }

    double* ET = launchTransposeCUDA(N,N,E);
    double* temp1 = launchMatMulCUDA(N,N,M,ET,D_T);
    double* VT = launchMatMulCUDA(M,N,M,SigmaInverse,temp1);


    ///////// Checking correctness
    // double* t1 = launchMatMulCUDA(N,M,M,Sigma,VT);
    // printf("Matmul4\n");
    // double* chec = launchMatMulCUDA(N,N,M,E,t1);
    // printf("Check below for correctness\n");
    // // printMat(N,N,E);

    // int count = 0;
    // for(int i=0;i<N;i++){
    //     for(int j=0;j<M;j++){
    //         if(isEqualDouble(D_T[i*M+j],chec[i*M+j])==false){
    //             count++;
    //         }
    //     }
    // }

    // printf("Number of incorrect values in SVD : %d\n",count);

    ///////////
    /////////////////////////////////////////////////////
    /////////// PCA 

    double sum = 0;
    double total_sum = 0;
    int k=1;
    for(int i=0;i<N;i++){
      total_sum += e[i];
    }
    for(int i=0;i<N;i++){
      sum+= e[i]/total_sum;
        if((100*sum)>=retention){
            break;
        }
        k++;
    }

    double* projectionMatrix = (double*) malloc(sizeof(double)*k*N);

    //TODO : Possible parallelizable scope
    for(int i=0;i<k;i++){
        for(int j=0;j<N;j++){
            projectionMatrix[j*k+i] = E[j*N+i];
        }
    }

    // printf("M : %d , N : %d , K : %d \n",M,N,k);
    double* DHAT = get_Dhat(D,M,N,k,projectionMatrix);
    // check_matmul_correctness(DHAT,M,N,k,D,projectionMatrix);


    ////////////////////////////////////////////////////////////
    ////////////Setting the pointers////////////////////////////
    *D_HAT = DHAT;
    *K = k;
    *V_T = VT;
    *SIGMA = sigma_new;
    *U = E;
    *SIGMAm = N;
    *SIGMAn = M;
}

