

import numpy as np
import matplotlib.pyplot as plt


def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  


def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m)) 
        centroids[i,:] = dataSet[index,:]
    return centroids


def KMeans(dataSet,k):
    m = np.shape(dataSet)[0]  
  
    clusterAssment = np.mat(np.zeros((m,2))) 
    clusterChange = True
   
    centroids = np.array([
        [6.2,3.2],
        [6.6,3.7],
        [6.5,3.0]
    ])

    iteration = 0 
    while clusterChange:
        clusterChange = False
        print('The ' + str(iteration) + ' iteration,' + "print point,center and distance:")
      
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            
            for j in range(k):
                
                distance = distEclud(centroids[j,:],dataSet[i,:])
               
                print(dataSet[i], centroids[j, :], distance)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
           
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2



       
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]] 
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   
            print("The new center", j, centroids[j,:])

        iteration = iteration + 1
        print() 

    print("Congratulations,cluster complete!")
    return centroids,clusterAssment

def showCluster(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    if n != 2:
        print("The data is not two-dimensional")
        return 1
    mark = ['or', 'og', 'ob', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("K is too big")
        return 1

    
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])
    mark = ['Dr', 'Dg', 'Db', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i])
    plt.show()

dataSet = np.array([
    [5.5,4.2],
    [5.1,3.8],
    [6.6,3.7],
    [4.7,3.2],
    [5.9,3.2],
    [6.2,3.2],
    [4.9,3.1],
    [6.7,3.1],
    [5.0, 3.0],
    [6.0, 3.0],
    [6.5, 3.0],
    [4.6, 2.9],
    [6.2, 2.8]
])
k = 3
centroids,clusterAssment = KMeans(dataSet,k)
showCluster(dataSet,k,centroids,clusterAssment)