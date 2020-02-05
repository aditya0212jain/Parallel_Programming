#include "lab4_mpi.h"

#include <malloc.h>
#include "mpi.h"

int Duel(char* Z,int Z_length,char* Y,int Y_length,int* phi,int phi_length,int i,int j)
{
  int k = phi[j-i];
  if(j+k >= Z_length){
    return i;
  }
  if(Z[j+k] != Y[k]){
    return i;
  }
  return j;
}

//Helper function #2
int getOverHeadBlock(int n,int nt,int i){
	i +=1;
	if(i<=n%nt){
		return n/nt+1;
	}else{
		return n/nt;
	}
}

char* getSubstring(char* S,int i,int l)
{
  /* printf("1\n"); */
  char* c = (char*)malloc(sizeof(char)*(l+1));
  for(int j=0;j<l;j++){
    *(c+j) = *(S+j+i);
  }
  *(c+l) = '\0';
  return c;
}

char* getStringConcat(char* a,int n,char* b,int m)
{
  char* ans = (char*)malloc(sizeof(char)*(n+m+1));
  for(int i=0;i<n;i++){
    *(ans + i) = *(a+i);
  }
  for(int j=n;j<(n+m);j++){
    *(ans+j) = *(b+j-n);
  }
  *(ans+n+m) = '\0';
  return ans;
}

int valueInArray(int* arr,int n,int k)
{
  for(int i=0;i<n;i++){
    if(arr[i]==k){
      return 1;
    }
  }
  return 0;
}

int matchPatternBruteForce(char* P,int P_m,char* T,int T_n,int i)
{
  if(i+P_m-1 >= T_n){
    return 0;
  }
  for(int j=i;j<i+P_m;j++){
    if(*(P+j-i) != *(T+j)){
      return 0;
    }
  }
  return 1;
}

void getSArray(int* M,int M_n,int i,int p,int** S,int* S_len){
  int count=0;
  for(int a =i;a<M_n;a+=p){
    count++;
  }
  int* temp = (int*)malloc(sizeof(int)*count);
  for(int a=0;a<count;a++){
    temp[a] = M[i+a*p];
  }
  *S = temp;
  *S_len = count;
}

int contains_consecutive_k(int* S,int S_len,int i,int k)
{
  if(i+k-1>=S_len){
    return 0;
  }
  int count =0;
  for(int a=i;a<i+k;a++){
    if(S[a]!=1){
      return 0;
    }
  }
  return 1;
}

int* getWitness(char* P,int P_m,int p)
{
  int* phi = (int*) malloc(sizeof(int)*p);
  phi[0] = 0;
  int flag1=1,flag2=1;
  for(int i=1;i<p;i++){
    flag2 = 1;
    for(int k=0;k<P_m-i && flag2 ==1;k++){
      if(P[k]!=P[k+i]){
	 *(phi+i) = k;
	flag2 = 0;
      }
    }
  }
  return phi;
}


void NP_TextAnalysis(char* T,int T_n,char* P,int P_m,int* phi,int phi_length,int** pos,int* pos_length)
{
  int mby2 = P_m%2 == 0 ? P_m/2 : (P_m/2 + 1) ;
  printf("mby2 : %d\n",mby2);
  int b = T_n%mby2==0 ? (T_n/mby2 - 1 ) : (T_n/mby2)  ;
  printf("b : %d\n",b);
  b++;
  char* Tb[b];
  int Tb_len[b];
  int cumulative_sum_Tb[b];
  printf("check1\n");
  for(int i=0;i<b;i++){
    Tb_len[i] = getOverHeadBlock(T_n,b,i);
    if(i==0){
      cumulative_sum_Tb[i] = 0;
    }else{
      cumulative_sum_Tb[i] = cumulative_sum_Tb[i-1]+Tb_len[i];
    }
  }
  
  for(int i=0;i<b;i++){
    Tb[i] = (char *)malloc(sizeof(char)*(Tb_len[i]+1));
    for(int j=0;j<Tb_len[i] && cumulative_sum_Tb[i]+j <T_n;j++){
      *(Tb[i]+j) = *(T+cumulative_sum_Tb[i]+j);
    }
    *(Tb[i]+Tb_len[i]) = '\0';
  }
  for(int i=0;i<b;i++){
    printf("cumulative[%d] : %d ",i,cumulative_sum_Tb[i]);
    printf("Tb[%d] : %s %d\n",i,Tb[i],Tb_len[i]);
  }

  int* potential_positions = (int*)malloc(sizeof(int)*b);
  for(int bi=0;bi<b;bi++){
    int i = cumulative_sum_Tb[bi] - Tb_len[bi];
    for(int j=i+1;j<cumulative_sum_Tb[bi];j++){//Possible error 
      i = Duel(T,T_n,P,P_m,phi,phi_length,i,j);
    }
    potential_positions[bi] = i;
  }

  int count=0;
  int match[b];

  for(int i=0;i<b;i++){
    int k = potential_positions[i];
    if(matchPatternBruteForce(P,P_m,T,T_n,k)==1){
      match[count] = k;
      count++;
    }
  }
  int* match_positions = (int*)malloc(sizeof(int)*count);
  for(int i=0;i<count;i++){
    match_positions[i] = match[i];
  }
  *pos = match_positions;
  *pos_length = count;
}



// p is the period of the pattern P
int* P_TextAnalysis(char* T,int T_n,char* P,int P_m,int p,int* match_length){
  char* p_dash = getSubstring(P,0,2*p-1);
  printf("p_dash : %s\n",p_dash); 
  int* phi = getWitness(p_dash,2*p-1,p);
  printf("phi : ");
  for(int i=0;i<p;i++){
    printf("%d ",phi[i]);
  }
  printf("\n");
  int* pos;
  int pos_length;
  NP_TextAnalysis(T,T_n,p_dash,2*p-1,phi,p,&pos,&pos_length);
  printf("pos_lenght : %d pos: ",pos_length);
  for(int i=0;i<pos_length;i++){
    printf("%d ",pos[i]);
  }
  printf("\n");
  char* u = getSubstring(P,0,p);
  int k = P_m/p;
  char* v = getSubstring(P,k*p,P_m-(k*p));
  printf("u : %s , k : %d , v: %s \n",u,k,v);
  int* M = (int*) malloc(sizeof(int)*T_n);
  //u^2v
  char* tempu2v = getStringConcat(u,p,u,p);
  char* u2v = getStringConcat(tempu2v,2*p,v,P_m-(k*p));
  printf("temp: %s u2v : %s  \n",tempu2v,u2v);
  for(int i=0;i<T_n;i++){
    M[i] = 0;
    if(valueInArray(pos,pos_length,i)==1){
      if(matchPatternBruteForce(u2v,2*p+P_m-(k*p),T,T_n,i)==1){
  	M[i] = 1;
      }
    }
  }
  printf("M : ");
  for(int i=0;i<T_n;i++){
    printf("%d ",M[i]);
  }
  printf("\n");
  int* S[p];
  int S_len[p];
  int* C[p];
  for(int i=0;i<p;i++){
    getSArray(M,T_n,i,p,&S[i],&S_len[i]);
    C[i] = (int*) malloc(sizeof(int)*S_len[i]);
    for(int j=0;j<S_len[i];j++){
      *(C[i]+j) = 0;
      if( contains_consecutive_k(S[i],S_len[i],j,k-1) == 1 ){
  	*(C[i]+j) = 1;
      }
    }
  }
  for(int i=0;i<p;i++){
    printf("S[%d]: ",i);
    for(int j=0;j<S_len[i];j++){
      printf("%d ",S[i][j]);
    }
    printf("\n");
     printf("C[%d]: ",i);
    for(int j=0;j<S_len[i];j++){
      printf("%d ",C[i][j]);
    }
    printf("\n");
  }
  
  int* MATCHES = (int*) malloc(sizeof(int)*(T_n-P_m+1));
  for(int j=0;j<T_n-P_m+1;j++){
    int i = j%p;
    int l = j/p;
    *(MATCHES+j) = *(C[i] + l);
  }
  *match_length = T_n - P_m +1;
  return MATCHES;
}
// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void periodic_pattern_matching (
		int n, 
		char *text, 
		int num_patterns, 
		int *m_set, 
		int *p_set, 
		char **pattern_set, 
		int **match_counts, 
		int **matches)
{
  int ans_length=0;
  printf("m_set : %d  p_set : %d , pattern : %s ",m_set[0],p_set[0],pattern_set[0]);
  int* ans = P_TextAnalysis(text,n,pattern_set[0],m_set[0],p_set[0],&ans_length);
  for(int i=0;i<ans_length;i++){
    printf("%d ",ans[i]);
  }
  printf("\n");
  
  
}
