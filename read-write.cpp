#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>
#define READER 5
#define WRITER 3 
using namespace std;
int readcont=0;
int writercont=0;
sem_t cont;
sem_t write1;
pthread_t rid[READER];
pthread_t wid[WRITER];
void *Reader(void*arg){
    int read =*((int*)arg);
    while(true){
    	  sem_wait(&queue);    //queue
    	  
        sem_wait(&cont);
        readcont++;
        if(readcont==1){
           sem_wait(&write1);
        }
        sem_post(&cont);
        
        sem_post(&queue);     //
        
        cout<<"reader"<<read<<" is reading"<<endl;
        sleep(1);
        
        
        sem_wait(&cont);
        readcont--;
        cout<<"reader"<<read<<" is over"<<endl;
        if(readcont==0){
          sem_post(&write1);
        }
        sem_post(&cont);
        
        
        sleep(2);
    }
}
void *Writer(void*arg){
    int writer =*((int*)arg);
    while(true){
    	
    	  sem_wait(&cont);
        writercont++;
        if(writercont==1){
           sem_wait(&queue);    //
        }
        sem_post(&cont);
    	
    	
       sem_wait(&write1);
       cout<<"writer"<<writer<<" is writing"<<endl;
       sleep(1);
       cout<<"writer"<<writer<<" is over"<<endl;
       sem_post(&write1);
       
        sem_wait(&cont);
        writercont--;
        if(writercont==0){
          sem_post(&queue);     //
        }
        sem_post(&cont);
       
       
       sleep(1);
    }
}

int main(){
   int a[5]={1,2,3,4,5};
   sem_init(&cont,0,1);
   sem_init(&write1,0,1);
   sem_init(&queue,0,1);
   for(int i=0; i<READER; i++){
      pthread_create(&rid[i],NULL,Reader,&a[i]);
   }
   for(int i=0; i<WRITER; i++){
      pthread_create(&wid[i],NULL,Writer,&a[i]);
   }
   for(int i=0; i<READER; i++){
      pthread_join(rid[i],NULL);
   }
   for(int i=0; i<WRITER; i++){
      pthread_join(wid[i],NULL);
   }
   return 0;
}