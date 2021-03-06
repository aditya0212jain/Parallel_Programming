#include <iostream>
#include <bits/stdc++.h>
#include "lab1_sequential.h"
#include "lab1_io.h"

#define lli float
#define fori(i,n) for(int i=0;i<n;i++)

using namespace std;

class KMeans{
	int n;
	const int d = 3;
	int k;
	lli **centroids;
	int *clusterToWhichItBelongs;
	lli **data;
	lli* centroidReturn;
	
	public:
	int numberOfInterations=25;
	int iterations=0;

	KMeans(int n,int k,int *data_points){
		//initializing parameters
		this->n = n;
		this->k = k;
		this->centroids = new lli* [k];
		this->clusterToWhichItBelongs = new int[this->n];
		this->data = new lli* [n];
		//Below one is for return to the main function
		this->centroidReturn = new lli[this->d*this->k*numberOfInterations];
		
		fori(i,this->n){
			this->data[i] = new lli[this->d];
			this->clusterToWhichItBelongs[i] = -1;
			fori(j,this->d){
				//loading data into our object's data
				this->data[i][j] = data_points[i*this->d + j];
			} 
		}
		fori(i,this->k){
			this->centroids[i] = new lli[this->d];
			fori(j,this->d){
				this->centroids[i][j] = j;
			}
		}
			
	}

	lli distance(lli *dataPoint,int cI){
		lli dist =0;
		fori(i,this->d){
			lli temp = dataPoint[i] - this->centroids[cI][i];
			temp *= temp;
			dist +=temp;
		}
		return (lli)sqrt(dist);
	}

	void printICentroids(){
		cout<<"Initial Centroids Sequential"<<endl;
		fori(i,this->k){
			fori(j,this->d){
				cout<<this->centroids[i][j]<<" ";
			}
			cout<<endl;
		}
	}

	void initializeCentroids(){
		//choose one then D(x) then sample one using D(X)^2 probaibility distribution
		// srand ( time(NULL) );
		// int r = rand()%this->n;
		// fori(i,this->d){
		// 	this->centroids[0][i] = this->data[r][i];
		// }
		// int assigned = 1;
		// double dx[this->n];
		// double max_value=0;
		// int max_index =r;
		// double total_sum = 0;
		// while(assigned<this->k){
		// 	total_sum = 0;
		// 	max_value = 0;
		// 	fori(i,this->n){
		// 		double min = INFINITY;
		// 		fori(j,assigned){
		// 			double temp = this->distance(this->data[i],j);
		// 			if(temp < min){
		// 				dx[i] = temp*temp;
		// 			}
		// 		}
		// 		total_sum += dx[i];
		// 		if(max_value<dx[i]){
		// 			max_value = dx[i];
		// 			max_index = i;
		// 		}
		// 	}
		// 	fori(i,this->d){
		// 		this->centroids[assigned][i] = this->data[max_index][i];
		// 	}
		// 	assigned++;
		// }

		// srand ( time(NULL) );
		bool visited[this->n]={0};
		fori(i,this->k){
			int r = rand()%this->n;
			while(visited[r]){
				r = rand()%this->n;
			}
			visited[r] = true;
			fori(j,this->d){		
				this->centroids[i][j] = this->data[r][j];
			}
		}
		// this->printICentroids();

	}

	void updateClass(){
		fori(i,this->n){
			lli minDistance = INFINITY;
			fori(j,this->k){
				lli tempDistance = this->distance(this->data[i],j);
				if(tempDistance < minDistance){
					this->clusterToWhichItBelongs[i] = j;
					minDistance = tempDistance;
				}
			}
		}
	}

	bool updateCentroids(){
		bool converg = false;
		lli sumOfClusterData[this->k][this->d] = {0};
		int numberOfDataPointsInCluster[this->k] = {0};
		fori(i,this->n){
			int label = clusterToWhichItBelongs[i];
			numberOfDataPointsInCluster[label] += 1;
			fori(j,this->d){
				sumOfClusterData[label][j] += this->data[i][j];
			}
		}
		lli sum = 0;
		lli total_sum = 0;
		fori(i,this->k){
			if(numberOfDataPointsInCluster[i]!=0){
				fori(j,this->d){
					sumOfClusterData[i][j] = (lli)sumOfClusterData[i][j]/numberOfDataPointsInCluster[i];
					///this is the stopping code
					total_sum += (lli)abs(this->centroids[i][j]);
					sum += (lli)abs(sumOfClusterData[i][j]-this->centroids[i][j]);
					///
					this->centroids[i][j] = sumOfClusterData[i][j];
					this->centroidReturn[iterations*(this->k*this->d)+this->d*i+j] = this->centroids[i][j];
				}
			}
		}
		if(sum<(3*total_sum/100)){
			converg = true;
		}
		return converg;
	}

	void run(){
		bool convergence = false;
		this->initializeCentroids();
		iterations =0;
		while(!convergence || iterations < numberOfInterations){//(!convergence && iterations < numberOfInterations)
			this->updateClass();
			convergence = this->updateCentroids();
			iterations += 1 ;
		}
		cout<<"Iterations "<<iterations<<endl;
	}

	float* getCentroid(){
		return this->centroidReturn;
	}

	int* getDataCluster(){
		int* temp = new int[this->n*(this->d+1)];
		fori(i,this->n){
			fori(j,this->d){
				temp[i*(this->d+1)+j] = this->data[i][j];
			}
			temp[i*(this->d+1)+this->d] = this->clusterToWhichItBelongs[i];
		}
		return temp;
	}

};

void kmeans_sequential(int N,
					int K,
					int* data_points,
					int** data_point_cluster,
					float** centroids,
					int* num_iterations
					){
		printf("Sequential\n");
		
		KMeans* a = new KMeans(N,K,data_points);
		a->run();
		*num_iterations = a->iterations - 1 ;
		*centroids = a->getCentroid();
		*data_point_cluster = a->getDataCluster();
		
	}