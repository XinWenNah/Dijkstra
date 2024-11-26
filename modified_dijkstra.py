import math
class MinHeap:
    """
    It will return the minimum element when the get_min function is called
    """
    MIN_CAPACITY = 1

    def __init__(self, max_size):
        """
	    Function description: This constrcutor will create an array base on the given size
	
	    Input: 
		max_size: The length of the array
		
		Output: None
		
		Time complexity: O(n), where n is the max_size 
		Aux space complexity: O(n),  where n is the length of the inner_array
        Space complexity: O(n), where n is the length of the inner_array
	    """
        self.length = 0
        self.inner_array = [math.inf] * (max(self.MIN_CAPACITY, max_size) + 1)

    def __len__(self):
        """
	    Function description: This function will return the current size of minHeap
	
	    Input: None
		
		Output: The length of the inner array
		
		Time complexity: O(1)
		Aux space complexity: O(1)
         Space complexity: O(1)
	    """
        return self.length

    def is_full(self) :
        """
	    Function description: This function will return True if the inner array is full else False
	
	    Input: None
		
		Output: None
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity: O(1)
	    """
        return self.length + 1 == len(self.inner_array)

    def rise(self, number):
        """
        Function description: This function will rise element at index k to its correct position.
	
	    Input: 
            number: The index of the item in the inner array, it should be 1 <= k <= self.length
		
		Output: None
		
		Time complexity: O(log n), where n is the number of item of inner array
		Aux space complexity: O(1)
        Space complexity: O(1)
        """
        item = self.inner_array[ number]
        while  number > 1 and item < self.inner_array[ number // 2]:
            self.inner_array[ number] = self.inner_array[ number // 2]
            self.inner_array[ number // 2].index =  number
            number =  number // 2
        self.inner_array[ number] = item
        item.index =  number

    def add(self, element):
        """
        Function description: This function will add a new element into minHeap and rise it to correct position.
	
	    Input: 
            element: The new element that is going to be added to the minHeap
		
		Output: None
		
		Time complexity: O(log n), where n is the number of item of inner array
		Aux space complexity: O(1)
        Space complexity: O(1)
        """
        if self.is_full():
            raise IndexError("Heap is full")

        self.length += 1
        self.inner_array[self.length] = element
        self.rise(self.length)#O(log n)

    def smallest_child(self, number):
        """
        Function description: This function will return the index of k's child with the smallest value.
	
	    Input: 
            number: The index of the parent, it should be 1 <= k <= self.length // 2
		
		Output: None
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity: O(1)
        """
        if 2 * number == self.length or self.inner_array[2 * number] < self.inner_array[2 * number + 1]:
            return 2 * number
        else:
            return 2 * number + 1

    def sink(self, number):
        """
        Function description: This function will make the element at index number sink to the correct position.
	
	    Input: 
            number: The index of the element, it should be 1 <= k <= self.length
		
		Output: None
		
		Time complexity: O(log n) , where n is the number of items in inner array
		Aux space complexity: O(1)
        Space complexity: O(1)
        """
        item = self.inner_array[number]

        while 2 * number <= self.length:
            min_child = self.smallest_child(number)
            if self.inner_array[min_child] >= item:
                break
            self.inner_array[number] = self.inner_array[min_child]
            self.inner_array[min_child].index = number
            number = min_child

        self.inner_array[number] = item
        item.index = number

    def get_min(self):
        """
        Function description: This function will remove (and return) the minimum element from the heap
	
	    Input: None
		
		Output: None
		
		Time complexity: O(log n) , where n is the number of items in inner array
		Aux space complexity: O(1)
        Space complexity: O(1)
        """
        if self.length == 0:
            raise IndexError("Heap is empty")

        min_elt = self.inner_array[1]
        self.inner_array[1] = self.inner_array[self.length]
        self.length -= 1
        if self.length > 0:
            self.sink(1)
        return min_elt
    
    def update_distance(self, location, new_distance):
        """
        Function description: This function will update the distance of the item, then rise it to correct position
	
	    Input: 
            location: The item
            new_distance: The new distance for the item
		
		Output: None
		
		Time complexity: O(log n) , where n is the number of items in inner array
		Aux space complexity: O(1)
        Space complexity: O(1)
        """
        self.inner_array[location.index].distance = new_distance
        self.rise(location.index)#O(log n)


class Vertex:
    """
    Vertex class
    This class object will store Edge class objects and other properties
    """
    def __init__(self, id, num):
        self.id = id
        self.edges_list = []
        self.visited = False
        self.discovered = False
        self.distance = math.inf
        self.previous = None
        self.index = None
        self.friends = None
        self.original_friends = None
        self.graph_num = num#Show that this vertex belongs to which graph, 0 belongs to g1, 1 belongs to g2
    
       
    def add_road(self, new_road):
        """
        Function description: This function add the new_road to edges_list property 
	
	    Input: 
            new_road: The Edge class object
		
		Output: None
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        self.edges_list.append(new_road)

    def get_id(self):
        """
        Function description: This function will return number identifier of itself 
	
	    Input: None
		
		Output: None
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        return int(self.id)
    
    def get_edges(self):
        """
        Function description: This function add the new_road to edges_list property 
	
	    Input: None
		
		Output: edges_list, A list that stores all the edges the vertex has
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        return self.edges_list
    
   
    def decide_friend(self, new_friend):
        """
        Function description: This function assign the new_friend to friends property if self.friends is None
                                if the vertex has friends, it will choose the lesser required station, else first come first serve 
	
	    Input: new_friend, a tuple (f, r), f is the friend's name, r is the required station
		
		Output: None
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        if(self.friends == None or new_friend[1] < self.friends[1]):
            self.friends = new_friend
            self.original_friends = new_friend
        

    def __eq__(self, other):
        """
        Function description: This function check the other.distance equals to self.distance or not, and return the result
	
	    Input: other, Other vertex
		
		Output: boolean
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        if(other is None):
            return False
        return self.distance == other.distance
    
    def __gt__(self, other):
        """
        Function description: This function check the self.distance greater than other.distance or not, and return the result
	
	    Input: other, Other vertex
		
		Output: boolean
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        if(other is None):
            return False
        return self.distance > other.distance
    
    def __lt__(self, other):
        """
        Function description: This function check the self.distance lesser than other.distance or not, and return the result
	
	    Input: other, Other vertex
		
		Output: boolean
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        if(other is None):
            return False
        return self.distance < other.distance
    

    def __le__(self, other):
        """
        Function description: This function check the self.distance lesser or equals to other.distance or not, and return the result
	
	    Input: other, Other vertex
		
		Output: boolean
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        if(other is None):
            return False
        return self.distance <= other.distance
    
    def __ge__(self, other):
        """
        Function description: This function check the self.distance greater or equals to other.distance or not, and return the result
	
	    Input: other, Other vertex
		
		Output: boolean
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        if(other is None):
            return False
        return self.distance >= other.distance


class Edge:
    """
    Edge class
    It stores the start, end and weight
    """
    def __init__(self, start, end, value):
        self.start = start
        self.end = end
        self.value = value
        


    
class CityMap:
    """
    CityMap class

    A graph data structure, which has plan function to get the shortest path to fecth he friend and go to destination.
    """
    def __init__(self, roads, tracks, friends):
        """
        Function description: This constructor will preprocess the arguments.

		Approach description: 
        My approach is creating two graphs, g1 has friends while g2 has no friends.
        When the dijkstra algorithm finds friend in g1, it will teleport to g2, ensures that we will have friend when we reach the destination.
        
        My preprocess will be creating two arrays, each array will store the same vertices and edges.
        After that, looping through the friends to assign friend to the g1.
        Looping through the tracks to assign those friends who can travel to different vertices and ensure each vertex has maximum one friend.
        When two friends can go to the same vertex, we choose the friend needs lesser stations. If both of them are same, then first come first serve.
        
		
		Input: 
			roads: An array of tuples ,(u,v,w) u is the start, v is the end, w is the weight
                    It is undirected, u can go to v while v can go to u.

            tracks: An array of tuples ,(u,v,w) u is the start, v is the end, w is the weight
                    It is directed, only u can go to v.

            friends: An array of tuples, (f, v) f is the friend's name, v is the identifier of vertex they stay
		
		Time complexity: O(R + T), where R is the number of roads, T is the number of tracks 

        Time complexity analysis: 

            Given that R is the number of roads, L is the number of locations, T is the number of tracks, F is the number of friends
            Looping through the roads to get the maximum identifier number needs O(R) time complexity,

            creating two arrays to store all the vertices cost O(2L) time complexity

            Looping through the roads to create the vertices and edges needs O(R) time complexity. 

            Looping through the friends list to assign the friend to initial vertex needs O(F) time complexity

            Looping through the tracks to assign those moveable friends to different vertices needs O(T) time complexity.

            O(R)+O(R)+O(F)+O(T)+O(2L) = O(R + F + T + 2L)
            Since each vertex can store maximum one friend, F = L, O(R + F + T + 2L) = O(R + L + T + 2L) = O(R + 3L + T )
            and R is significantly smaller than L^2, but R greater or equal to L-1(tree),else the graph is disconnected, so O(R) >= O(L)
            Therefore, the time complexity is O(R + T)


		Aux space complexity: O(R),  where R is the number of roads

        Aux space complexity analysis:

            Given that R is the number of roads, L is the number of locations, T is the number of tracks, F is the number of friends
            Creating two arrays to store all the vertices cost O(2L) aux space complexity 
 
            Since we need to create same edge for g1 and g2, and create edges which just swap the destination and start(undirected), 
            so we will create each edge 4 times, it needs O(4R) aux space complexity

            Since each vertex can store maximum one friend in its property, aux space complexity is O(1)

            O(2L)+O(4R)+ O(1) = O(2L + 4R), since O(R) >= O(L), it becomes O(4R) = O(R)

        Space complexity:O(n)+O(R) = O(r+t+f+R) , where r,t and f refers to the input size of roads, tracks and friends, R is the aux space complexity
        """

        #Loop through the roads to find maximum identifier number
        #Time:O(R)
        #Space:O(1)
        max_size = 0
        for r in roads:#O(R)
            for i in range(2):#O(1)
                if(r[i] > max_size):#O(1)
                    max_size = r[i]#O(1)

        
        #Create two arrays, length is the maximum number + 1, representing two graphs
        self.with_f = [None]*(max_size+1)
        self.without_f = [None]*(max_size+1)
        
        #Loop through the roads to create the Edge class object and Vertex class object
        #Add the Edge object to the Vertex object accordingly.
        #After that, assign that Vertex object to the two arrays base on their identifier number
        for road in roads:#O(R)
            for i in range(2):#O(1) 
                if(self.with_f[road[i]] == None):
                    new_location = Vertex(road[i], 0)
                    new_location.add_road(Edge(road[0], road[1], road[2]))
                    new_location.add_road(Edge(road[1], road[0], road[2]))
                    self.with_f[road[i]] = new_location

                    new_location1 = Vertex(road[i], 1)
                    new_location1.add_road(Edge(road[0], road[1], road[2]))
                    new_location1.add_road(Edge(road[1], road[0], road[2]))
                    self.without_f[road[i]] = new_location1
                else:
                    self.with_f[road[i]].add_road(Edge(road[0], road[1], road[2]))   
                    self.with_f[road[i]].add_road(Edge(road[1], road[0], road[2])) 
                    self.without_f[road[i]].add_road(Edge(road[0], road[1], road[2]))   
                    self.without_f[road[i]].add_road(Edge(road[1], road[0], road[2]))   
        
        #Loop through the friends 
        #Assign the tuples,(f, n) to the initial vertex, where f means friend's name, n means the required station 
        for f in friends:
            self.with_f[f[1]].decide_friend((f[0], 0))

        #Loop through the tracks to get the next station
        #If current station has friend, assign the friend to the next station and increase the number of required station
        #If the required station is greater than 1, he can't move to the next station, we skip 
        for t in tracks:
            current_friend = self.with_f[t[0]].friends
            if(current_friend == None or current_friend[1] > 1):
                continue
            self.with_f[t[1]].decide_friend((current_friend[0], current_friend[1]+1))
        
        
        
    def reset_distance_friends(self):
        """
        Function description: This function resets the vertices of the two arrays to default(discovered = False,visited = False,...). 
	
        Input:None
            
        Output: None
            
        Time complexity: O(L), where L is the number of locations
        Aux space complexity: O(1)
        Space complexity: O(1)
        """
        for i in range(len(self.with_f)):#O(L)
            location0 = self.with_f[i] 
            location1 = self.without_f[i]

            location0.distance = math.inf
            location0.friends = location0.original_friends
            location0.discovered = False
            location0.visited = False
            location0.previous = None
            location0.index = None 

            location1.distance = math.inf
            location1.friends = location1.original_friends
            location1.discovered = False
            location1.visited = False
            location1.previous = None
            location1.index = None


    def check_got_friend(self, index):
        """
        Function description: This function returns True when the vertex in with_f array has friends, else return False 
	
	    Input: The identifier number of the vertex
		
		Output: boolean 
		
		Time complexity: O(1)
		Aux space complexity: O(1)
        Space complexity:O(1)
        """
        return self.with_f[index].friends != None
    

    def plan( self, start, destination):
        """
		Function description: This function will implement the dijkstra algorithm to find out the shortest from to given start to given destination.After that, it will return a tuple contains the required time, route, friend and the vertex we fetch the friend

		Approach description: 
            Need to reset the all vertices to default first.
            Basically, we find the nearest friend in graph1, find the shortest path to destination in graph2.
            Since we want to find the shortest distance to every vertex, I will implement dijkstra algorithm and minHeap to help me to achieve it.
            minHeap will always return the nearest vertex, and dijkstra algorithm will take it and moving to next location.
            When we find the nearest friend in g1, I will bring it to graph2 to the same vertex.
            It can ensure that we always have a friend when we reach the destination in graph2, because in graph2, we start with friend.

            While exploring the graph2, we will meet other friend, we need to decide to dsicard the current friend or ignore the new one.
            If the new friend has shorter distance, we pick them.If they have the same distance as our current friend's, we check who has lesser required station.
            If both of their's required stations is identical, then we ignore the new friend.
            When we finally find the destination in graph2, we will stop appending the vertex to our route, but we will not terminate.
            Because there may be better friends, we will keep running and updating until we have explored all vertices.
            After that, we will do back tracking for the route to get the true route



		
		Input: 
			start: An identifier number of vertex , it should inside the range of 0 until L-1, where L is the number of vertices

            destination : An identifier number of vertex , it should inside the range of 0 until L-1,  where L is the number of vertices
		
		Time complexity: O(R log L), where R is the number of roads, L is the number of locations 
        
        Time compelxity annalysis:
                
                Given that R is the number of roads, L is the number of locations, T is the number of tracks, F is the number of friends


                Resetting the graph1 and graph2 costs O(L) time complexity,

                Creating the minHeap needs O(2L) time complexity,

                while loop will loop 2R time, time complexity is O(R)

                get_min() needs O(log L) time complexity,

                minHeap.add() needs O(log L) time complexity

                time compelexity of looping through the all the edges of the vertex is lesser than O(R),because R is significantly smaller than L^2, 
                show that the graph is not dense, which means not every vertex has all the edges,so the looping time will smaller than R times

                Inside the looping of edges, we need to call minHeap.add(), needs O(log L) time complexity

                Backtracking needs O(R) time complexity

                Reverse needs O(R) time complexity too

                O(L) + O(2L) + O(R * (log L + log L + logL)) + O(4R) + O(R) + O(R) 
                = O(3L + RlogL + 6R)
                Since O(R) >= O(L)
                = O(6R + RlogL)
                = O(RlogL)

		Aux space complexity: O(L), where L is the number of locations
         
        Aux space complexity analysis:
                Creating the minHeap costs O(2L) aux space complexity,

                Appending things to route(store all vertices) needs O(L) aux space complexity,
                
                Backtracking and appending vertex to other route needs O(L) aux space complexity,

                O(2L) + O(L) +O(L) = O(4L) = O(L)

		Space complexity: O(n+R), where n is the user input, L is the number of locations
	    """
        #Reset everything to default
        self.reset_distance_friends()#O(L)

        
        minHeap = MinHeap(2*len(self.with_f))#O(2L)

        route = []
        time = 0
        pick_up_friend = None 
        
        current = self.with_f[start]
        current.discovered = True
        current.distance = 0

        #O(1), because current minHeap is empty, no need to rise or sink
        minHeap.add(current)

        #Used to decide when do we need to stop adding the location to our route
        stop_adding = False

        #In worst case, we need to go through all the vertices in two graphs.
        #So this while loop will run 2R time
        while(len(minHeap) > 0):

            #O(log L)
            current_location = minHeap.get_min()
            
            if(current_location.visited):
                continue

            current_location.visited = True

            #If the vertex in g1 has friends, need to bring that friend to g2
            if(self.check_got_friend(int(current_location.get_id()))):
                f = self.with_f[current_location.get_id()].friends
                
                #Get the same vertex in g2
                new_location = self.without_f[int(current_location.get_id())]

                #Since we just "teleport" to g2 from g1, the distance does not change
                new_location.distance = current_location.distance

                #If the vertex in g2 has friends, we need to decide which friend we want
                if(new_location.friends is not None):
                    #check distance and required station
                    #If the new_location.friends needs more time, or its required station is higher, choose the current_location.friends
                    if(new_location.friends[3] > current_location.distance or (new_location.friends[3] == current_location.distance and new_location.friends[2] > f[1])):
                        new_location.friends = (f[0], current_location.get_id(), f[1], current_location.distance)   

                else:#If new location has no friends
                    new_location.friends = (f[0], current_location.get_id(), f[1], current_location.distance)
                    new_location.previous = current_location.previous
                new_location.discovered = True

                minHeap.add(new_location)#O(log L)

            #If we have a super vertex, which means it is connected to all vertices,
            #Time complexity will be O(R)
            #However, this situation may only happen few times, because specs mention |R| is significantly smaller than |L^2|

            for e in current_location.get_edges():

                #Get the correct vertex, do not mix them together
                next_location = self.without_f[e.end]
                if(current_location.graph_num == 0):#graph with friend
                    next_location = self.with_f[e.end]
                    
                if(not next_location.discovered):
                    next_location.discovered = True
                    next_location.distance = current_location.distance + e.value
                    next_location.previous = current_location
                    
                    #If the vertex in graph2, if the current location has friends but the next location has no friend, we need to assign the friend to next location ith different distance
                    #Because we may bring the friend to that location
                    if(current_location.graph_num == 1 and next_location.friends is None and current_location.friends is not None):
                        next_location.friends = (current_location.friends[0],current_location.friends[1], current_location.friends[2], next_location.distance)

                    minHeap.add(next_location)

                elif(not next_location.visited):
                    original_distance = next_location.distance
                    new_distance = min(next_location.distance, current_location.distance + e.value)
                    minHeap.update_distance(next_location, new_distance)

                    #If new distance not equals to original distance, means we choose different location, 
                    #So the we need to assign new location to the next_location.previous
                    if(new_distance != original_distance):
                        next_location.previous = current_location

                    #If in graph2, and current location has friend, we will bring it to next location with different distance
                    if(current_location.graph_num == 1 and current_location.friends is not None):
                        #However, the discovered next location may have friend, so we need to decide which friend we want to pick
                        if(next_location.friends is None or new_distance != original_distance or (new_distance == original_distance and next_location.friends[2] > current_location.friends[2])):
                            next_location.friends = (current_location.friends[0],current_location.friends[1], current_location.friends[2], new_distance)
                            next_location.previous = current_location
                                    
            #e will keep adding the vertex to route if we havent found the destination in GRAPH2, not graph1           
            if(not stop_adding):
                time = current_location.distance
                route.append(current_location)
                pick_up_friend = current_location.friends
                stop_adding = self.without_f[destination].visited#Check the destination in graph2 is visited or not
            #Although we have visited the destination, it does not mean we can terminate, there may be other better routes, so we still need to run
                
        #Back tracking
        #The last vertex will always be our destination
        current = route[-1]
        return_route = [current.get_id()]
        #If we have only one friend, and that friend is far away from our start and destination
        #And the graph is a tree, which means we need to go through every vertex, the time complexity will be O(2R) = O(R)
        while(current.previous != None):
            current = current.previous
            return_route.append(current.get_id())
        
        #Time complexity can be O(R), if our route stores all vertices
        return_route.reverse()
        friend_name = pick_up_friend[0]
        vertex_of_friend  = int(pick_up_friend[1])
        return(time, return_route, friend_name, vertex_of_friend)  
        
        
