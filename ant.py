import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import inf


def ant(distance,n):
    iteration = 10
    beta = 2
    alpha = 0.1
    evaporation = 0.5
    m = n
    visibility = 1/distance
    visibility[visibility == inf] = 0

    pheromne = .1*np.ones((m,n))

    tour = np.zeros((m,n+1))

    print(tour)

    for k in range(iteration):
        tour[:,0] = 1

        for i in range(m):
            temp_visibility = np.array(visibility)

            for j in range(n-1):
                features = np.zeros(n)
                cum_prob = np.zeros(n)

                cur_loc = int(tour[i,j]-1)  
                temp_visibility[:,cur_loc] = 0     #making visibility of the current city as zero
            
                p_feature = np.power(pheromne[cur_loc,:],beta)         #calculating pheromne feature 
                v_feature = np.power(temp_visibility[cur_loc,:],alpha)

                p_feature = p_feature[:,np.newaxis]                     #adding axis to make a size[5,1]
                v_feature = v_feature[:,np.newaxis]                     #adding axis to make a size[5,1]
            
                feature = np.multiply(p_feature,v_feature)

                total = np.sum(feature)

                probs = feature/total 

                m = max(probs)

                for i in range(len(probs)):
                    if probs[i] == m:
                        m = i
                city = probs[m]

                tour[i,j+1] = city+1
            
            left = list(set([i for i in range(1,n+1)])-set(tour[i,:-2]))[0]     #finding the last untraversed city to route
        
            tour[i,-2] = left

        rute_opt = np.array(tour)               #intializing optimal route
    
        dist_cost = np.zeros((m,1)) 

        for i in range(m):
        
            s = 0
            for j in range(n-1):
            
                s = s + distance[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1]   #calcualting total tour distance
        
            dist_cost[i]=s 

        dist_min_loc = np.argmin(dist_cost)             #finding location of minimum of dist_cost
        dist_min_cost = dist_cost[dist_min_loc]         #finging min of dist_cost
    
        best_route = tour[dist_min_loc,:]               #intializing current traversed as best route
        pheromne = (1-evaporation)*pheromne                       #evaporation of pheromne with (1-e)
    
        for i in range(m):
            for j in range(n-1):
                dt = 1/dist_cost[i]
                pheromne[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1] +=  dt   

    print('route of all the ants at the end :')
    print(rute_opt)
    print()
    print('best path :',best_route)
    print('cost of the best path',int(dist_min_cost[0]) + distance[int(best_route[-2])-1,0])

G = nx.Graph()
color = []
no_of_cities = int(input("Enter the total number of cities : "))
cities = [i for i in range(1,no_of_cities + 1)]
G.add_nodes_from(cities)
color.append('green')
distance = np.zeros((no_of_cities,no_of_cities))
ch = 'y'
print("Let's add edges to the graph  ")
while(ch == 'y'):
    start = int(input("Enter the starting vertex : "))
    end = int(input("Enter the end vertex : "))
    weight = float(input("Enter the weight on the edge : "))
    G.add_edge(start,end,weight = weight)
    distance[start-1][end-1] = weight
    distance[end-1][start-1] = weight
    ch = input("Do you want to add more edges : ")

nx.draw(G,with_labels = True,node_color = color,node_size = 1000)
pos = nx.spectral_layout(G)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size=15)
plt.show()
print(distance)
ant(distance,no_of_cities)