#include<bits/stdc++.h>
using namespace std;

struct t{
    int** temp;
};
void change(t po){
    int *cl;
    cl = *po.temp;
    cl[0]= 5;
}

int main(){
    int* tem;
    int p[5];
    tem = &p[0];
    for(int i=0;i<5;i++){
        p[i]=i;
    }
    struct t po;
    po.temp = &tem;
    change(po);
    for(int i=0;i<5;i++){
        cout<<p[i]<<endl;
    }
    
    return 0;
}