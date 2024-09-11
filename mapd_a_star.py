import os
from collections import deque
from queue import PriorityQueue
import os
import heapq
from collections import defaultdict

'''
--- Environment ---

A. read input function
B. Package, Agent, and Saboteur classes
C. State class: Contains functions to move agents and to uopdate accordingly
'''
def read_input(input_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists
    packages = []
    blocked_edges = []
    agents = []
    saboteurs = []
    fragile_edges = []
    fragile_edges_objects = []

    for line in lines:
        if line.startswith('#X'):
            x = int(line.split()[1])
        elif line.startswith('#Y'):
            y = int(line.split()[1])

        elif line.startswith('#P'):
            package_details = []
            elements = line.split()
            for element in elements:
                try:
                    package_details.append(int(element))
                except ValueError:
                    pass
            package = Package(package_details)
            packages.append(package)

        elif line.startswith('#B'):
            elements = line.split()
            elements = [int(element) for element in elements[1:]]
            blocked_edge = ((elements[0], elements[1]), (elements[2], elements[3]))
            blocked_edge_reversed = ((elements[2], elements[3]), (elements[0], elements[1]))
            blocked_edges.extend([blocked_edge, blocked_edge_reversed])
        
        elif line.startswith('#F'):
            elements = line.split()
            elements = [int(element) for element in elements[1:]]
            f_edge = ((elements[0], elements[1]), (elements[2], elements[3]))
            f_edge_reversed = ((elements[2], elements[3]), (elements[0], elements[1]))
            fragile_edges.extend([f_edge, f_edge_reversed])
            fragile_edges_objects.append(Fragile_edge((elements[0], elements[1]), (elements[2], elements[3])))

        elif line.startswith('#A'):
            elements = line.split()
            agent_x = int(elements[1])
            agent_y = int(elements[2])
            agent_num = len(agents) + 1
            agent = Agent(agent_x, agent_y, agent_num)
            agents.append(agent)
        
        elif line.startswith('#S'):
            elements = line.split()
            saboteur_x = int(elements[1])
            saboteur_y = int(elements[2])
            saboteur_num = len(saboteurs) + 1
            saboteur = Saboteur(saboteur_x, saboteur_y, saboteur_num)
            saboteurs.append(saboteur)
        
    return(x, y, fragile_edges_objects, packages, agents, saboteurs, blocked_edges, fragile_edges)



def build_grid(blocked_edges = [], blocked_edges_2 = []):
    blocked_edges.extend(blocked_edges_2)
    graph = {}
    for x in range(x_limit + 1):
        for y in range(y_limit + 1):
            graph[(x,y)] = []
    
    for x in range(x_limit + 1):
        for y in range(y_limit + 1):
            if x+1 <= x_limit:  graph[(x,y)].append((x+1,y))
            if x-1 >= 0:        graph[(x,y)].append((x-1,y))
            if y+1 <= y_limit:  graph[(x,y)].append((x,y+1))
            if y-1 >= 0:        graph[(x,y)].append((x,y-1))
        
    for edge in blocked_edges:
        vertex_1 = edge[0]
        vertex_2 = edge[1]
        if vertex_2 in graph[vertex_1]: graph[vertex_1].remove(vertex_2)
        if vertex_1 in graph[vertex_2]: graph[vertex_2].remove(vertex_1)
    
    return graph


class Fragile_edge:
    def __init__(self, vertex_1, vertex_2):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.broken = False

class Package:
    def __init__(self, details):
        self.pickup_location = (details[0], details[1])
        self.pickup_time = details[2]
        self.delivery_location = (details[3], details[4])
        self.delivery_time = (details[5])
        self.picked_up = False
        self.delivered = False
        self.agent = -1
    
    def copy(self):
        package_copy = Package([0,0,0,0,0,0])
        package_copy.pickup_location =      self.pickup_location
        package_copy.pickup_time =          self.pickup_time
        package_copy.delivery_location =    self.delivery_location
        package_copy.delivery_time =        self.delivery_time
        package_copy.picked_up =            self.picked_up
        package_copy.delivered =            self.delivered
        package_copy.agent =                self.agent
        return package_copy

class Agent:
    def __init__(self, x, y, num):
        self.location = (x,y)
        self.score = 0
        self.num = num
    
    def copy(self):
        agent_copy = Agent(0,0,0)
        agent_copy.location = self.location
        agent_copy.score = self.score
        agent_copy.num = self.num
        return agent_copy


class Saboteur:
    def __init__(self,x,y,num):
        self.location = (x,y)
        self.score = 0
        self.num = num
    
    def copy(self):
        saboteur_copy = Saboteur(0,0,self.num)
        saboteur_copy.location = self.location
        saboteur_copy.score = self.score
        return saboteur_copy

class State:
    def __init__(self, packages, agents, saboteurs, blocked_edges, fragile_edges, path, time = 0, g = 0, h = -1, f = -1):
        self.g = g
        self.h = h
        self.f = f
        self.blocked_edges = blocked_edges
        self.fragile_edges = fragile_edges
        self.time = time
        self.grid = build_grid(blocked_edges)
        self.currently_used_edges = []

        self.agents = []
        for agent in agents: self.agents.append(agent.copy())
        self.saboteurs = []
        for saboteur in saboteurs: self.saboteurs.append(saboteur.copy())
        self.packages = []
        for package in packages: self.packages.append(package.copy())

        self.path = [] 
        for loc in path: self.path.append(loc)
        if self.path == []: self.path.append(self.agents[0].location)

        self.update(self.agents[0].location, self.agents[0].location, self.agents[0])
    
    def __lt__(self, other):
            return self.f < other.f
    
    def copy(self):
        state_copy = State(self.packages, self.agents, self.saboteurs, self.blocked_edges, self.fragile_edges, self.path, self.g, self.h, self.f)
        return state_copy
    
    def move_agent(self, agent, new_loc):
        old_loc = agent.location

        if (new_loc in self.grid[old_loc] and 
            (new_loc, old_loc) not in self.currently_used_edges and
            not any(new_loc == agent.location for agent in self.agents) ):
            agent.location = new_loc
        
        self.update(old_loc, new_loc, agent)
    
    def update(self, old_loc, new_loc, agent):
        self.currently_used_edges.extend([(old_loc, new_loc), (new_loc, old_loc)])

        for package in self.packages:
            # pick up available packages
            if(not package.picked_up and not package.delivered and package.pickup_time <= self.time):
                if agent.location == package.pickup_location:
                    package.picked_up = True
                    package.agent = agent.num

            # deliver relevant packages 
            if(package.picked_up and not package.delivered and package.agent == agent.num and package.delivery_time >= self.time):
                if agent.location == package.delivery_location:
                    package.delivered = True
                    package.agent = -1
                    agent.score += 1
            
            # break relevant fragile edges
            if (new_loc, old_loc) in self.fragile_edges:
                self.blocked_edges.extend([(old_loc, new_loc), (new_loc, old_loc)])
        
    def move_saboteur(self, saboteur, new_loc):
        old_loc = saboteur.location

        if new_loc in self.grid[old_loc] and ((new_loc, old_loc) not in self.currently_used_edges):
            saboteur.location = new_loc
        
        self.currently_used_edges.extend([(old_loc, new_loc), (new_loc, old_loc)])

        if (new_loc, old_loc) in self.fragile_edges:
            self.blocked_edges.extend([(old_loc, new_loc), (new_loc, old_loc)])
            saboteur.score += 1

        if saboteur.num == len(self.saboteurs) : self.round_update()
    
    def round_update(self):
        self.time += 1
        self.grid = build_grid(blocked_edges)
        self.currently_used_edges = []


'''
--- Part 1 ---
A. Stupid greedy
B. Saboteur move
C. Shortest path (BFS)
'''

def stupid_greedy(agent, state):
    move = None
    min_length = float('inf')
    min_pacakge = None

    # agent got packages
    for package in state.packages:
        if package.agent == agent.num and package.delivery_time >= state.time:
            path, path_length = shortest_path(agent.location, package.delivery_location, state.grid)
            if path_length < min_length:
                move = path[1]
                min_length = path_length
                min_package = package
            # tie breaker
            if path_length == min_length:
                if (min_package and (package.delivery_location[0] < min_package.delivery_location[0] or
                                     (package.delivery_location[0] == min_package.delivery_location[0] and 
                                      package.delivery_location[1] < min_package.delivery_location[1]))):
                    move = path[1]
                    min_package = package

    if move: return move

    # agent doesnt have packages
    for package in state.packages:
        if not package.picked_up and not package.delivered and package.pickup_time <= state.time:
            path, path_length = shortest_path(agent.location, package.pickup_location, state.grid)
            if path_length < min_length:
                if path_length <= 2: continue
                else: move = path[1]
                min_length = path_length
                min_package = package
            # tie breaker
            if path_length == min_length:
                if (min_package and (package.pickup_location[0] < min_package.pickup_location[0] or
                                        (package.pickup_location[0] == min_package.pickup_location[0] and 
                                        package.pickup_location[1] < min_package.pickup_location[1]))):
                    move = path[1]
                    min_package = package
    
    if move:
        return move
    else:
        return agent.location

def saboteur_move(saboteur, state):
    move = saboteur.location
    min_length = float('inf')
    goal = None

    for edge in fragile_edges_objects:
        if not edge.broken:
            # if at a fragile edge, break it
            if edge.vertex_1 == saboteur.location:
                edge.broken = True
                saboteur.location = edge.vertex_2
                saboteur.score += 1
                return edge.vertex_2

            if edge.vertex_2 == saboteur.location:
                edge.broken = True
                saboteur.location = edge.vertex_1
                saboteur.score += 1
                return edge.vertex_1
                        
            # else, go to closest fragile edge
            path_1, path_length_1 = shortest_path(saboteur.location, edge.vertex_1, state.grid)
            path, path_length = shortest_path(saboteur.location, edge.vertex_2, state.grid)
            vertex = edge.vertex_2
            if path_length_1 < path_length:
                path = path_1
                path_length = path_length_1
                vertex = edge.vertex_1 
            if path_length < min_length:
                move = path[1]
                goal = vertex
                min_length = path_length
            # tie breaker
            if path_length == min_length:
                if (goal and (vertex[0] < goal[0] or (vertex[0] == goal[0] and vertex[1] < goal[1]))):
                    move = path[1]
                    goal = vertex
    return move

# BFS
def shortest_path(start, end, grid):
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        current_vertex, path = queue.popleft()
        
        if current_vertex == end:
            return path, len(path) - 1  # Return path and its length
        
        if current_vertex not in visited:
            visited.add(current_vertex)
            
            for neighbor in grid[current_vertex]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    # If no path is found
    return None, float('inf')



'''
--- Part 2 ---
A. greedy search
B. A*
C. real time A*
'''

## A*
def A_star_move(state, limit = 10000):
    state_0 = state.copy()
    state_0.h = h_score(state_0)
    state_0.g = 0
    state_0.f = state_0.h + state_0.g + state_0.agents[0].location[0]*0.1 + state_0.agents[0].location[0]*0.01

    states = PriorityQueue()
    states.put(state_0)
    num_expands = 0
    best_state = None
    min_f = float('inf')
    done = False
    while num_expands < limit:
        current_state = states.get()

        if all(package.delivered for package in current_state.packages):
            done = True
            best_state = current_state
            break
        
        if current_state.f <= min_f and len(current_state.path) >= 2: #and current_state.path[0] != current_state.path[1]:
            min_f = current_state.f
            best_state = current_state

        new_states = expand(current_state)
        num_expands += 1
        [states.put(new_state) for new_state in new_states]

    return best_state, num_expands, done

def expand(cur_state):
    cur_loc = cur_state.agents[0].location
    moves = [cur_loc, (cur_loc[0]+1, cur_loc[1]), (cur_loc[0]-1, cur_loc[1]), (cur_loc[0], cur_loc[1]+1), (cur_loc[0], cur_loc[1]-1)]
    new_states = []
    for move in moves:
        new_state = cur_state.copy()
        new_state.move_agent(new_state.agents[0], move)
        new_state.round_update()
        new_state.h = h_score(new_state)
        new_state.g = cur_state.g + 1
        new_state.path.append(new_state.agents[0].location)
        new_state.f = new_state.h + new_state.g + new_state.agents[0].location[0]*0.1 + new_state.agents[0].location[0]*0.01

        new_states.append(new_state)
    
    return new_states



def h_score(state):
    graph = state.grid
    relevant_vertices = [state.agents[0].location]
    for package in state.packages:
        if package.picked_up == False:
            #and package.pickup_time <= state.time
            relevant_vertices.append(package.pickup_location)
        if package.delivered == False:
            #and package.delivery_time >= state.time
            relevant_vertices.append(package.delivery_location)

    if len({vertice for vertice in relevant_vertices}) == 1: return 0
    weighted_graph = build_weighted_graph(graph, relevant_vertices)
    mst = prim(weighted_graph)
    h = calculate_mst_weight(mst)
    return h

def dijkstra(graph, start):
    # Initialize distances to all vertices as infinity
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    
    # Initialize a priority queue with the start vertex
    pq = [(0, start)]  # (distance, vertex)
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        # Skip processing if this path is no longer the shortest to this vertex
        if current_distance > distances[current_vertex]:
            continue
            
        # Visit each neighbor of the current vertex
        for neighbor in graph[current_vertex]:
            distance = current_distance + 1
            # Update distance if shorter path found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

def build_weighted_graph(graph, special_vertices):
    weighted_graph = defaultdict(list)
    
    # Calculate shortest paths from each relevant vertex to all other special vertices
    for vertex in special_vertices:
        shortest_paths = dijkstra(graph, vertex)
        # Add shortest paths to the weighted graph
        for destination, distance in shortest_paths.items():
            if destination in special_vertices and vertex != destination:
                weighted_graph[vertex].append((destination, distance))
    
    return weighted_graph

def prim(graph):
    mst = {}
    visited = set()
    pq = [(0, None, next(iter(graph)))]  # (weight, source, destination)

    while pq:
        # Get the minimum weight edge from the priority queue
        weight, source, destination = heapq.heappop(pq)
        
        # Check if the destination vertex has been visited
        if destination not in visited:
            # Add the edge to the minimum spanning tree
            if source is not None:
                mst[source] = mst.get(source, []) + [(destination, weight)]
            # Mark the destination vertex as visited
            visited.add(destination)
            # Add the neighboring edges of the destination vertex to the priority queue
            for neighbor, neighbor_weight in graph[destination]:
                if neighbor not in visited:
                    heapq.heappush(pq, (neighbor_weight, destination, neighbor))

    return mst

def calculate_mst_weight(mst):
    total_weight = 0
    for vertex, edges in mst.items():
        for neighbor, weight in edges:
            total_weight += weight
    return total_weight

            

    

            



    
#### Main
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'input1.txt')                       # change file path here if needed!
x_limit, y_limit, fragile_edges_objects, packages, agents, saboteurs, blocked_edges, fragile_edges = read_input(file_path)
state = State(packages, agents, saboteurs, blocked_edges, fragile_edges, [])

#### Part 1:
#### Stupid greedy and saboteur
print("\nPart 1:")
steps = 10
agent_path = [state.agents[0].location]
saboteur_path = [state.saboteurs[0].location]
for step in range(steps):
    for agent in state.agents:
        move = stupid_greedy(agent, state)
        state.move_agent(agent, move)
        agent_path.append(state.agents[0].location)
    for saboteur in state.saboteurs:
        move = saboteur_move(saboteur, state)
        state.move_saboteur(saboteur, move)
        saboteur_path.append(state.saboteurs[0].location)

print(f"sabotuer: edges broken: {state.saboteurs[0].score}, path: {saboteur_path}")
print(f"agent: score: {state.agents[0].score}, path: {agent_path}")



#### Part 2: 
#### Greedy search, A*, and real time A*
print("\nPart 2: Single agent")

# reading input
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'input2.txt')                       # change file path here if needed!
x_limit, y_limit, fragile_edges_objects, packages, agents, saboteurs, blocked_edges, fragile_edges = read_input(file_path)
state = State(packages, agents, saboteurs, blocked_edges, fragile_edges, [])


# A*
final_state, expands, done = A_star_move(state)
print("1. A*:")
if done: print("Found a solution")
else: print("Didnt find a solution")
print(f"Path: {final_state.path}")
print(f"Score: {final_state.agents[0].score}")
print(f"Expands: {expands}")


# real time A*
x_limit, y_limit, fragile_edges_objects, packages, agents, saboteurs, blocked_edges, fragile_edges = read_input(file_path)
state = State(packages, agents, saboteurs, blocked_edges, fragile_edges, [])
steps = 15

path = [state.agents[0].location]
total_expands = 0
solved = False
for step in range(steps):
    if all(package.delivered for package in state.packages):
        break
    solution, expands, done = A_star_move(state, 1000)
    if done: 
        solved = True
    if len(solution.path)<2:
        break
    total_expands += expands
    state.move_agent(state.agents[0], solution.path[1])
    state.round_update()
    path.append(state.agents[0].location)

print("\n2. Real time A*:")
if state.agents[0].score == len(state.packages): print("Found a solution")
else: print("Didnt find a solution")
print(f"Path: {path}")
print(f"Score: {state.agents[0].score}")
print(f"Expands: {total_expands}")

# Greedy search
def greedy_path(state):
    state_0 = state.copy()
    state_0.h = h_score(state_0)
    state_0.g = 0
    state_0.f = state_0.h + state_0.g + state_0.agents[0].location[0]*0.1 + state_0.agents[0].location[0]*0.01

    min_f = float('inf')
    cur_loc = state_0.agents[0].location
    best_move = (cur_loc[0]+1, cur_loc[1])
    moves = [(cur_loc[0]+1, cur_loc[1]), (cur_loc[0]-1, cur_loc[1]), (cur_loc[0], cur_loc[1]+1), (cur_loc[0], cur_loc[1]-1)]
    for move in moves:
        new_state = state_0.copy()
        new_state.move_agent(new_state.agents[0], move)
        cur_h = h_score(new_state)
        cur_f = cur_h + new_state.agents[0].location[0]*0.1 + new_state.agents[0].location[0]*0.01
        if cur_f < min_f:
            min_f = cur_f
            best_move = move
    
    return best_move


x_limit, y_limit, fragile_edges_objects, packages, agents, saboteurs, blocked_edges, fragile_edges = read_input(file_path)
state = State(packages, agents, saboteurs, blocked_edges, fragile_edges, [])
steps = 3

path = [state.agents[0].location]
total_expands = 0
solved = False
for step in range(steps):
    if all(package.delivered for package in state.packages):
        solved = True
        break
    move = greedy_path(state)
    state.round_update()
    total_expands += 1
    state.move_agent(state.agents[0], move)
    state.round_update()
    path.append(state.agents[0].location)

print("\n2. Greedy search:")
if solved: print("Found a solution")
else: print("Didnt find a solution")
print(f"Path: {path}")
print(f"Score: {state.agents[0].score}")
print(f"Expands: {total_expands}")