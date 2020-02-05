#include <iostream>
#include <bits/stdc++.h>
#include<pthread.h>
#include<vector>
#include "lab1_pthread.h"
#include "lab1_io.h"

#define lli float
#define fori(i,n) for(int i=0;i<n;i++)

using namespace std;

// pthread_mutex_t lock1; 

//Helper function #1
int getStartingForI(int n,int nt,int i){
	int divtemp = n/nt;
	int remtemp = n%nt;
	i +=1;
	if(i<=remtemp){
		return (divtemp+1)*(i-1);
	}else{
		return (divtemp)*(i-1) + remtemp;
	}
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


//Structure of thread argument passed
struct thread_data {
   int  thread_id;
   int starting_point;
   int ending_point;
   void *object;
};

/*
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
												kMeans STARTING HERE
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
*/

class KMeans{
	int n;
	const int d = 3;
	int k;
	lli **centroids;
	int *clusterToWhichItBelongs;
	lli **data;
	lli* centroidReturn;
	
	public:

	bool convergence=false;
	int numberOfIterations = 25;
	int iterations = 0;
	lli** sumOfClusterData;//[k][d] 
	int* numberOfDataPointsInCluster;//[k]

	KMeans(int n,int k,int* data_points){
		this->n = n;
		this->k = k;
		this->centroids = new lli* [k];
		this->clusterToWhichItBelongs = new int[this->n];
		this->sumOfClusterData = new lli* [k];
		this->numberOfDataPointsInCluster = new int[this->k];
		this->data = new lli* [n];
		this->centroidReturn = new lli[this->d*this->k*(numberOfIterations)];

		fori(i,this->n){
			this->data[i] = new lli[this->d];
			this->clusterToWhichItBelongs[i] = -1;
			fori(j,this->d){
				this->data[i][j] = data_points[i*this->d + j];
			} 
		}
		fori(i,this->k){
			this->centroids[i] = new lli[this->d];
			fori(j,this->d){
				this->centroids[i][j] = j;
			}
		}

		fori(i,this->k){
			sumOfClusterData[i] = new lli[this->d];
			numberOfDataPointsInCluster[i] = 0;
		}
		fori(i,this->k){
			fori(j,this->d){
				this->sumOfClusterData[i][j] = 0;
			}
		}
		
	}

	void readyFirst(){
		// sumOfClusterData = new lli* [this->k];
		// numberOfDataPointsInCluster = new int[this->k];
		fori(i,this->k){
			// sumOfClusterData[i] = new lli[this->d];
			numberOfDataPointsInCluster[i] = 0;
			fori(j,this->d){
				this->sumOfClusterData[i][j] = 0;
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
		cout<<"Initial Centroids Pthread"<<endl;
		fori(i,this->k){
			fori(j,this->d){
				cout<<this->centroids[i][j]<<" ";
			}
			cout<<endl;
		}
	}

	void initializeCentroids(){
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

	static void *updateClass(void *threadarg){
		struct thread_data *my_data;
   		my_data = (struct thread_data *) threadarg;
		KMeans* tempObj;
		tempObj = (KMeans *)my_data->object;
		for(int i=my_data->starting_point;i<=my_data->ending_point;i++){
			lli minDistance = INFINITY;
			fori(j,tempObj->k){
				lli tempDistance = tempObj->distance(tempObj->data[i],j);
				if(tempDistance < minDistance){
					tempObj->clusterToWhichItBelongs[i] = j;
					minDistance = tempDistance;
				}
			}
		}
	}

	static void *updateHelpers(void *threadarg){
		struct thread_data *my_data;
   		my_data = (struct thread_data *) threadarg;
		KMeans* tempObj;
		tempObj = (KMeans *)my_data->object;
		for(int i=my_data->starting_point;i<=my_data->ending_point;i++){
			int label = tempObj->clusterToWhichItBelongs[i];
			tempObj->numberOfDataPointsInCluster[label] += 1;
			fori(j,tempObj->d){
				tempObj->sumOfClusterData[label][j] += tempObj->data[i][j];
			}
		}
	}

	static void *updateCentroids(void *threadarg){
		struct thread_data *my_data;
   		my_data = (struct thread_data *) threadarg;
		KMeans* tempObj;
		tempObj = (KMeans *)my_data->object;
		bool converg = false;
		lli sum = 0;
		lli total_sum = 0;

		for(int i=my_data->starting_point;i<=my_data->ending_point;i++){
			int label = tempObj->clusterToWhichItBelongs[i];
			tempObj->numberOfDataPointsInCluster[label] += 1;
			fori(j,tempObj->d){
				tempObj->sumOfClusterData[label][j] += tempObj->data[i][j];
			}
		}

		fori(i,tempObj->k){
			if(tempObj->numberOfDataPointsInCluster[i]!=0){
				fori(j,tempObj->d){
					tempObj->sumOfClusterData[i][j] = (lli)tempObj->sumOfClusterData[i][j]/tempObj->numberOfDataPointsInCluster[i];
					///this is the stopping code
					total_sum += (lli)abs(tempObj->centroids[i][j]);
					sum += (lli)abs(tempObj->sumOfClusterData[i][j]-tempObj->centroids[i][j]);
					///
					tempObj->centroids[i][j] = tempObj->sumOfClusterData[i][j];
					tempObj->centroidReturn[tempObj->iterations*(tempObj->k*tempObj->d)+tempObj->d*i+j] = tempObj->centroids[i][j];
				}
			}
		}
		if(sum<(3*total_sum/100)){
			converg = true;
		}
		tempObj->convergence = converg;

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




/*
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
												MAIN STARTING HERE
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
*/



void kmeans_pthread(int num_threads,
					int N,
					int K,
					int* data_points,
					int** data_point_cluster,
					float** centroids,
					int* num_iterations
					){

	// printf("PThreads\n");
	KMeans *ourObject = new KMeans(N,K,data_points);
	pthread_t threads[num_threads];
    struct thread_data td[num_threads];
   
    int rc;
    int i;

	/*
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
												RUN KMEANS									
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	*/

	////implementing the classification method here in main for better control over threads:
	bool convergence = false;
	ourObject->initializeCentroids();
	ourObject->iterations =0;
	while(!convergence || ourObject->iterations < ourObject->numberOfIterations){  //!ourObject->convergence && 

		// cout<<"ourObject iteration: "<<ourObject->iterations<<"\n"<<endl;

		//doing first updateClass 
		fori(i,num_threads){
			td[i].thread_id = i;
			//here n_input
			int po = getStartingForI(N,num_threads,i);
			td[i].starting_point = po;
			td[i].ending_point = po+getOverHeadBlock(N,num_threads,i)-1;
			td[i].object = ourObject;
			// cout<<"td starting: "<<td[i].starting_point<<" ending: "<<td[i].ending_point<<endl;
			int rc = pthread_create(&threads[i],NULL,&KMeans::updateClass,(void *)&td[i]);
			if(rc){
				cout << "Error:unable to create thread," << rc << endl;
         		exit(-1);
			}
		}
		//waiting for the threads to complete after updating clusters
		for(int j=0;j<num_threads;j++){
    	   pthread_join(threads[j],NULL);
   		}

		ourObject->readyFirst();

		td[0].thread_id = 0;
		td[0].starting_point = 0;
		td[0].ending_point = N-1;
		td[0].object = ourObject;
		int tp = pthread_create(&threads[0],NULL,&KMeans::updateCentroids,(void *)&td[0]);
		if(tp){
			cout << "Error:unable to create thread," << rc << endl;
         	exit(-1);
		}
		pthread_join(threads[0],NULL);

		ourObject->iterations += 1 ;
		convergence = ourObject->convergence;
	}
	// cout<<"Total iterations: "<<ourObject->iterations<<endl;
	*num_iterations = ourObject->iterations - 1;
	*centroids = ourObject->getCentroid();
	*data_point_cluster = ourObject->getDataCluster();
}

/*
updateClass: assigns new cluster values to the points according to the centroids
updateCentroids : computes new centroid(mean) values acc to the cluster points
algo:
1)assigns clusters to the points based on their distances
2)compute the new mean of the points and update the centroids 
*/