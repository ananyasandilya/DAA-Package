import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

# function to implement Ant Colony Optimization 

def ant(G,distance,n):
    iteration = 100
    beta = 2
    alpha = 0.1
    evaporation = 0.5

    m = 4
    visible = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if distance[i][j] != 0:
                visible[i][j] = 1/distance[i][j]

    pheromne = .1*np.ones((m,n))
    tour = np.zeros((m,n+1),dtype='i')
    for i in range(m):
        num = random.randint(1,n)
        tour[i][0] = num
        tour[i][n] = num
    print(tour)

    for k in range(iteration):

        for i in range(m):
            temp_visible = np.array(visible)
            l = []
            for j in range(n-1): 
                current_location = int(tour[i,j]-1)
                l.append(current_location+1)
                temp_visible[:,current_location] = 0              
                p_feature = np.power(pheromne[current_location,:],beta)
                v_feature = np.power(temp_visible[current_location,:],alpha)

                p_feature = p_feature[:,np.newaxis] 
                v_feature = v_feature[:,np.newaxis] 

                features = np.zeros(n)
                cum_prob = np.zeros(n)

                feature = np.multiply(p_feature,v_feature)

                total = np.sum(feature)
                #print(total,feature)
                probs = feature/total

                #print(current_location+1,np.argmax(probs))
                tour[i][j+1] = np.argmax(probs) + 1
            #print(temp_visible)
            #print(l)
            left = list(set([i for i in range(1,n+1)])-set(tour[i,:-2]))[0]     #finding the last untraversed city to route
        
            tour[i,-2] = left
        #print(tour)
        #print(probs[1])    

        rute_opt = np.array(tour)               #intializing optimal route
    
        dist_cost = np.zeros((m,1)) 

        for i in range(m):
           # print(distance)
            s = 0
           # print(rute_opt[i])
            for j in range(n):
            
                s = s + distance[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1]   #calcualting total tour distance
            print(s)
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
    print('cost of the best path',int(dist_min_cost[0]))
    return best_route

if __name__ == "__main__":
    
    G = nx.Graph()      
    color = []

    # Reading inputs from the file

    f = open("graph.txt",'r')
    input_graph = list(f.read().split('\n'))
    
    # Processing the input to add the nodes in the graph

    no_of_cities = int(input_graph[0])
    cities = [i for i in range(1,no_of_cities + 1)]
    G.add_nodes_from(cities)
    color.append('yellow')

    distance = np.zeros((no_of_cities,no_of_cities))

    # Processing the input to add the edges and weights to the graph

    print("Let's add edges to the graph  ")
    for i in range(1,len(input_graph)):
        start,end,weight = input_graph[i].split(' ')
        #print(start," ",end," ",weight)
        start = int(start)
        end = int(end)
        weight = float(weight)
        G.add_edge(start,end,weight = weight)
        distance[start-1][end-1] = weight
        distance[end-1][start-1] = weight

    nx.draw(G,with_labels = True,node_color = color,node_size = 1000,edge_color = "blue")
    pos = nx.spectral_layout(G)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size=15)
    plt.show()

    print("Distance metric of the graph : ")
    print(distance)
    ant(G,distance,no_of_cities)
    """path_edges = list(zip(best_route,best_route[1:]))
    print(path_edges)
    pos = nx.spectral_layout(G)
    nx.draw_networkx_nodes(G,pos,with_labels = True,node_color = color,node_size = 1000,edge_color = "blue")
    
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_size=15)
    nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)
    plt.axis('equal')
    plt.show()"""