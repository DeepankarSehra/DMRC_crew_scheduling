import csv
from datetime import timedelta
import networkx as nx
import random
import matplotlib.pyplot as plt
import inspect
import heapq
from collections import defaultdict
import gurobipy as gp 
from gurobipy import GRB
from collections import defaultdict
from collections import deque

class Service:
    def __init__(self, attrs):
        self.serv_num = int(attrs[0])
        self.train_num = attrs[1]
        self.start_stn = attrs[2]
        self.start_time = hhmm2mins(attrs[3])
        self.end_stn = attrs[4]
        self.end_time = hhmm2mins(attrs[5])
        self.direction = attrs[6]
        self.serv_dur = int(attrs[7])
        self.jurisdiction = attrs[8]
        self.stepback_train_num = attrs[9]
        self.serv_added = False
        self.break_dur = 0
        self.trip_dur = 0

def hhmm2mins(hhmm):
    h, m = map(int, hhmm.split(':'))
    return h*60 + m

def mins2hhmm(mins):
    h = mins // 60
    m = mins % 60
    return f"{h:02}:{m:02}"

def fetch_data(filename):
    services = []
    services_dict = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            serv_obj = Service(row)
            services.append(serv_obj)
            services_dict[serv_obj.serv_num] = serv_obj
    return services, services_dict

def fetch_data_by_rake(filename, partial=False, rakes=10):
    ''' Fetch data from the given CSV file '''
    services = []
    services_dict = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            serv_obj = Service(row)
            if partial:
                if serv_obj.train_num in [f"{700+i}" for i in range(rakes+1)]:
                    services.append(serv_obj)
                    services_dict[serv_obj.serv_num] = serv_obj
            else:
                services.append(serv_obj)
                services_dict[serv_obj.serv_num] = serv_obj
    return services, services_dict

def draw_graph_with_edges(graph, n=50):
    # Create a directed subgraph containing only the first n edges
    subgraph = nx.DiGraph()
    
    # Add the first n edges and associated nodes to the subgraph
    edge_count = 0
    for u, v in graph.edges():
        if edge_count >= n:
            break
        subgraph.add_edge(u, v)
        edge_count += 1

    # Plotting the directed subgraph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(subgraph)  # Position nodes using the spring layout
    nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color='red')
    nx.draw_networkx_labels(subgraph, pos, font_size=15)
    nx.draw_networkx_edges(
        subgraph, pos, arrowstyle='->', arrowsize=20, edge_color='blue'
    )
    
    plt.title(f"First {n} Directed Edges of the Network")
    # plt.show()
    plt.savefig(f'first{n}edges.png')

## checking if two services can be connected
def node_legal(service1, service2):
    if service1.stepback_train_num == "No Stepback":
        if service2.train_num == service1.train_num:
            if service1.end_stn == service2.start_stn and 0 <= (service2.start_time - service1.end_time) <= 15:
                return True
        else:
            if (service1.end_stn[:4] == service2.start_stn[:4]) and (service2.start_time >= service1.end_time + 30) and (service2.start_time <= service1.end_time + 150):
                return True
        
    else:
        if service2.train_num == service1.stepback_train_num:
            if (service1.end_stn == service2.start_stn) and (service1.end_time == service2.start_time):
                return True
        else:
            if (service1.end_stn[:4] == service2.start_stn[:4]) and (service2.start_time >= service1.end_time + 30 ) and (service2.start_time <= service1.end_time + 150):
                return True
    return False

def no_overlap(service1, service2):
    return service1.end_time <= service2.start_time

def count_overlaps(selected_duties, services):
    '''
    Checks the number of overlaps of services in selected_duties, and prints them

    Arguments: selected_duties - duties that are selected after column generation
               services - all services

    Returns: Boolean - False, if number of services != all services covered in selected_duties; else True
    '''
    services_covered = {}

    for service in services:
        services_covered[service.serv_num] = 0

    for duty in selected_duties:
        for service in duty:
            services_covered[service] += 1

    num_overlaps = 0
    num_services = 0
    for service in services_covered:
        if services_covered[service] > 1:
            num_overlaps += 1
        if services_covered[service] != 0:
            num_services += 1

    print(f"Number of duties selected: {len(selected_duties)}")
    print(f"Total number of services: {len(services)}")
    print(f"Number of services that overlap in duties: {num_overlaps}")
    print(f"Number of services covered in duties: {num_services}")

    if len(services) != num_services:
        return False
    else:
        return True

def create_duty_graph(services):
    G = nx.DiGraph()

    for i, service1 in enumerate(services):
        G.add_node(service1.serv_num)

    G.add_node(-1) #end_node
    G.add_node(-2) #start_node

    for i, service1 in enumerate(services):
        for j, service2 in enumerate(services):
            if i != j:
                if node_legal(service1, service2):
                    G.add_edge(service1.serv_num, service2.serv_num, weight=service1.serv_dur)


    #end node edges
    for i, service in enumerate(services):
        G.add_edge(service.serv_num, -1, weight=service.serv_dur)

    #start node edges
    for i, service in enumerate(services):
        G.add_edge(-2, service.serv_num, weight=0)
        
    return G

def extract_nodes(var_name):

    parts = var_name.split('_')
    if len(parts) != 3 or parts[0] != 'x':
        raise ValueError(f"Invalid variable name format: {var_name}")
    
    start_node = int(parts[1])
    end_node = int(parts[2])
    
    return start_node, end_node

def generate_paths(outgoing_var, show_paths = False):

    paths = []
    paths_decision_vars = []
    current = -2
    for start_path in outgoing_var[-2]:
        current_path = []
        current_path_decision_vars = []
        if start_path.x !=1:continue
        else:
            start, end = extract_nodes(start_path.VarName)
            # current_path.append(start_path.VarName)
            current_path.append(end)
            current_path_decision_vars.append(start_path)
            # start, end = extract_nodes(start_path.VarName)
            current = end
            while current != -1:
                for neighbour_edge in outgoing_var[current]:
                    if neighbour_edge.x !=1:continue
                    else:
                        start, end = extract_nodes(neighbour_edge.VarName)
                        current_path.append(end)
                        # current_path.append(neighbour_edge.VarName)
                        current_path_decision_vars.append(neighbour_edge)
                        # start, end = extract_nodes(neighbour_edge.VarName)
                        current = end
            paths.append(current_path)
            current_path.pop()
            paths_decision_vars.append(current_path_decision_vars)
            if show_paths:
                print(current_path)
    return paths, paths_decision_vars
def reconstruct_path(label):
            """
            Reconstruct the full path by backtracking through the 'pred' pointers.
            Each label is a tuple: (cost, resource_vector, node, pred).
            Returns the full path as a list of nodes.
            """
            path = []
            while label is not None:
                cost, resource, node, pred = label
                path.append(node)
                label = pred
            return list(reversed(path))
def solution_verify(services, duties, verbose =True):
    flag = True
    for service in services:
        service_check = False
        for duty in duties:
            if service.serv_num in duty:
                service_check = True
                break
        if service_check == False:
            if verbose:
                print(f"Service {service.serv_num} not assigned to any duty")
            flag= False
            break
    return flag

def roster_statistics(paths, service_dict):

    """
    service_dict: The dictionary of service times    
    """

    #1 Number of duties
    print("\nRoster Statistics:")
    print("Number of duties: ", len(paths))

    #2 Maximum number of services in a duty
    max_len_duty = 0
    min_len_duty = 1e9
    for duty in paths:
        if len(duty)>max_len_duty:
            max_len_duty = len(duty)
        if len(duty)<min_len_duty:
            min_len_duty = len(duty)

    print("Maximum number of services in a duty: ", max_len_duty-1)
    print("Minimum number of services in a duty: ", min_len_duty-1)

    #3 Maximum duration of a duty
    max_duration = 0
    min_duration = 1e9
    serv_dur_6 = 0
    serv_dur_8 = 0
    for duty in paths:
        current_duration = 0
        for service in duty:
            # start, end = extract_nodes(edge)
            if service != -2: current_duration += service_dict[service].serv_dur
        if current_duration > max_duration:
            max_duration = current_duration
        if current_duration < min_duration:
            min_duration = current_duration
        if current_duration > (6*60):
            serv_dur_6+=1
        if current_duration > (8*60):
            serv_dur_8+=1
            
    print("Maximum duration of duty: ", mins2hhmm(max_duration))
    print("Minimum duration of duty: ", mins2hhmm(min_duration))
    print("Duties with driving time more than 6hrs: ",  serv_dur_6)
    print("Duties with driving time more than 8hrs: ",  serv_dur_8)

def get_bad_paths(paths, paths_decision_vars, service_dict):
    bad_paths = []
    bad_paths_decision_vars = []
    for i in range(len(paths)):
        current_duration = 0
        for node in paths[i]:
            # start, end = extract_nodes(edge)
            # if node != -2: current_duration += service_dict[node].serv_dur
            if node != -2: current_duration += service_dict[node].serv_dur
        if current_duration > 6*60:
            bad_paths.append(paths[i])
            bad_paths_decision_vars.append(paths_decision_vars[i])

    return bad_paths, bad_paths_decision_vars

def get_lazy_constraints(bad_paths, bad_paths_decision_vars, service_dict):
    lazy_constraints = []
    for i in range(len(bad_paths)):
        current_duration = 0
        current_lazy_constr = []
        bad_paths[i].append(-2)  #to make the size of paths and path_decision_vars equal
        for j in range(len(bad_paths[i])):
            # start, end = extract_nodes(bad_paths[i][j])
            node = bad_paths[i][j]
            if node != -2: current_duration += service_dict[node].serv_dur
            current_lazy_constr.append(bad_paths_decision_vars[i][j])
            if current_duration > 6*60:
                lazy_constraints.append(current_lazy_constr)
                break

    return lazy_constraints



### Helpers for column generation
def can_append(duty, service):
    last_service = duty[-1]
    
    start_end_stn_tf = last_service.end_stn == service.start_stn
    # print(service.start_time, last_service.end_time)
    start_end_time_tf = 5 <= (service.start_time - last_service.end_time) <= 15
    start_end_stn_tf_after_break = last_service.end_stn[:4] == service.start_stn[:4]
    start_end_time_within = 50 <= (service.start_time - last_service.end_time) <= 150

    if last_service.stepback_train_num == "No StepBack":
        start_end_rake_tf = last_service.train_num == service.train_num
    else:
        start_end_rake_tf = last_service.stepback_train_num == service.train_num
    
    # Check for valid conditions and time limits
    if start_end_rake_tf and start_end_stn_tf and start_end_time_tf:
        time_dur = service.end_time - duty[0].start_time
        cont_time_dur = sum([serv.serv_dur for serv in duty])
        if cont_time_dur <= 180 and time_dur <= 445:
            return True
    elif start_end_time_within and start_end_stn_tf_after_break:
        time_dur = service.end_time - duty[0].start_time
        if time_dur <= 445:
            return True
    return False

def restricted_linear_program(service_dict, duties, show_solutions = False, show_objective = False, warm_start_solution=None, t=0):

    # objective = 0
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)

    
    ###Decision Variables
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=0, name=f"x{i}"))

    big_penalty = 1e6


    ### Objective
    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    
    ### Constraints
    service_constraints = []
    for service_idx, service in enumerate(service_dict.values()):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty)>= 1,
            # gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty)== 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)

    

    ### Warm Start from previous solution
    # if warm_start_solution:
    #     for i in warm_start_solution.keys():
    #         duty_vars[i].VBasis = gp.GRB.BASIC
    #         duty_vars[i].Start = warm_start_solution[i]

    # for v in model.getVars():
    #     if v.VBasis == gp.GRB.BASIC:
    #         # ct+=1
    #         # basis[int(v.VarNAME[1:])]= v.x
    #         print(f"Variable '{v.VarName}' is in the basis") 

    model.optimize()



    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
        return None, None, None, None
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        model.write("model.lp") 
        if show_solutions:
            print("Optimal solution found")
        
        # Get the dual variables for each service constraint
        # dual_values = [constr.Pi for constr in service_constraints]
        duals = dict([(constr.ConstrName, constr.Pi) for constr in service_constraints])

        
        solution = [v.x for v in model.getVars()]
        # print("Hi" ,len(solution))
        selected_duties = [(v.varName, v.x) for v in model.getVars() if v.x > 0]
        selected_duties_vars = [v for v in model.getVars() if v.x > 0]
        
        ct = 0 
        basis = {}
        for v in model.getVars():
            if v.VBasis == gp.GRB.BASIC:
                ct+=1
                basis[int(v.VarNAME[1:])]= v.x

                

        if show_solutions:
            print("Positive Duties, 0: ", len(selected_duties))
            for variable in selected_duties_vars:
                print(variable.varName, variable.x)
        if show_objective:    
            print(f"Objective Value: {objective.getValue()}")

        return objective.getValue(), duals, basis, selected_duties, selected_duties_vars
    else:
        print("No optimal solution found.")
        return None, None, None, None, None

def restricted_linear_program_for_heuristic(service_dict, duties, selected_vars, show_solutions = False, show_objective = False, warm_start_solution=None, t=0):

    # objective = 0
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)

    
    ###Decision Variables
    duty_vars = []
    for i in range(len(duties)):
        if i in selected_vars:
            duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=1, lb=1, name=f"x{i}"))
        else:
            duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=0, name=f"x{i}"))

    ### Objective
    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    ### Constraints
    service_constraints = []
    for service_idx, service in enumerate(service_dict.values()):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty)>= 1,
            # gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty)== 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)

    ### Warm Start from previous solution
    # if warm_start_solution:
    #     for i in warm_start_solution.keys():
    #         duty_vars[i].VBasis = gp.GRB.BASIC
    #         duty_vars[i].Start = warm_start_solution[i]

    # for v in model.getVars():
    #     if v.VBasis == gp.GRB.BASIC:
    #         # ct+=1
    #         # basis[int(v.VarNAME[1:])]= v.x
    #         print(f"Variable '{v.VarName}' is in the basis") 

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
        return None, None, None, None
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        model.write("model.lp") 
        if show_solutions:
            print("Optimal solution found")
        
        # Get the dual variables for each service constraint
        # dual_values = [constr.Pi for constr in service_constraints]
        duals = dict([(constr.ConstrName, constr.Pi) for constr in service_constraints])

        solution = [v.x for v in model.getVars()]
        # print("Hi" ,len(solution))
        selected_duties = [(v.varName, v.x) for v in model.getVars() if v.x > 0]
        selected_duties_vars = [v for v in model.getVars() if v.x > 0]
        
        ct = 0 
        basis = {}
        for v in model.getVars():
            if v.VBasis == gp.GRB.BASIC:
                ct+=1
                basis[int(v.VarNAME[1:])]= v.x

        if show_solutions:
            print("Positive Duties, 0: ", len(selected_duties))
            for variable in selected_duties_vars:
                print(variable.varName, variable.x)
        if show_objective:    
            print(f"Objective Value: {objective.getValue()}")

        return objective.getValue(), duals, basis, selected_duties, selected_duties_vars
    else:
        print("No optimal solution found.")
        return None, None, None, None, None

def generate_initial_feasible_duties_random_from_services(services, num_services, show_duties = False):

    feasible_duties = []

    # initial set of duties should cover all services
    # not checking for breaks
    for service1 in services:
        duty = [service1]
        for service2 in services:
            if service1.serv_num != service2.serv_num:
                if can_append(duty, service2):
                    duty.append(service2)
        feasible_duties.append(duty)

    # random_duties = random.sample(feasible_duties, num_services)
    serv_num_duty = []

    # to get duty in terms of service numbers
    for duty in feasible_duties:
        tt = []
        for serv in duty:
            tt.append(serv.serv_num)
        serv_num_duty.append(tt)
    if show_duties:
        print(serv_num_duty)
    return serv_num_duty


def generate_new_column(graph, service_dict, dual_values, method = "topological sort", verbose = False):
    

    if method == "topological sort":

        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 0
            else:
                # graph[u][v]['weight'] = dual_values[u]
                graph[u][v]['weight'] = dual_values["Service_" + str(u)]

        topo_order = list(nx.topological_sort(graph))

        longest_dist = {node: float('-inf') for node in graph.nodes}
        longest_dist[-2] = 0  # Distance to source is 0
        predecessor = {node: None for node in graph.nodes} 

        # Relax edges in topological order
        for u in topo_order:
            for v in graph.successors(u):  
                weight = graph[u][v].get('weight')  
                if longest_dist[v] < longest_dist[u] + weight:
                    longest_dist[v] = longest_dist[u] + weight
                    predecessor[v] = u

        
        shortest_path = []
        curr = -1

        while curr is not None:  
            shortest_path.append(curr)
            curr = predecessor[curr]
        # shortest_path.pop()
        shortest_path.reverse()  
        # shortest_path.pop()

        if verbose:
            path_duals = []
            # path_duals.append(graph[-2][shortest_path[0]]['weight'])
            for i in range(len(shortest_path)-1):
                path_duals.append(graph[shortest_path[i]][shortest_path[i+1]]['weight'])
            # path_duals.append(graph[shortest_path[-1]][-1]['weight'])

            print("Path Duals: ",path_duals)

        return shortest_path[1:-1], longest_dist[-1]
    
    elif method == "bellman ford":
        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 1
            else:
                # graph[u][v]['weight'] = -dual_values[u]
                graph[u][v]['weight'] = -dual_values["Service_" + str(u)]

        shortest_path = nx.shortest_path(graph, source=-2, target=-1, weight='weight', method = 'bellman-ford')
        shortest_distance = nx.shortest_path_length(graph, source=-2, target=-1, weight='weight', method = 'bellman-ford')

        if verbose:
            path_duals = []
            for i in range(len(shortest_path)-1):
                path_duals.append(graph[shortest_path[i]][shortest_path[i+1]]['weight'])

            print("Path Duals: ", path_duals)

        return shortest_path[1:-1], shortest_distance
                        


    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
def generate_new_column_2(graph, service_dict, dual_values, method = "topological sort", verbose = False, time_constr = 6*60):

    

    if method == "topological sort":

        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 0
            else:
                # graph[u][v]['weight'] = dual_values[u]
                graph[u][v]['weight'] = dual_values["Service_" + str(u)]

        topo_order = list(nx.topological_sort(graph))

        longest_dist = {node: float('-inf') for node in graph.nodes}
        longest_dist[-2] = 0  # Distance to source is 0
        duration = {node: 0 for node in graph.nodes}
        predecessor = {node: None for node in graph.nodes} 

        # Relax edges in topological order
        for u in topo_order:
            for v in graph.successors(u):  
                weight = graph[u][v].get('weight')
                if v!=-1:
                    succ_dur = service_dict[v].serv_dur
                else:
                    succ_dur = 0  
                if longest_dist[v] < longest_dist[u] + weight and duration[u] + succ_dur <= 6*60:
                    longest_dist[v] = longest_dist[u] + weight
                    duration[v] = duration[u] + succ_dur
                    predecessor[v] = u

        
        shortest_path = []
        curr = -1

        while curr is not None:  
            shortest_path.append(curr)
            curr = predecessor[curr]
        # shortest_path.pop()
        shortest_path.reverse()  
        # shortest_path.pop()

        if verbose:
            path_duals = []
            # path_duals.append(graph[-2][shortest_path[0]]['weight'])
            for i in range(len(shortest_path)-1):
                path_duals.append(graph[shortest_path[i]][shortest_path[i+1]]['weight'])
            # path_duals.append(graph[shortest_path[-1]][-1]['weight'])

            print("Path Duals: ",path_duals)

        return shortest_path[1:-1], longest_dist[-1]
    # //////////////////////////////////////////////////////////
    elif method == "bf_duration_constr":
       # Create a copy of the graph and adjust edge weights based on dual values.
        graph_copy = graph.copy()
        # sink_edge_weights = []
        print("bf_duration_constr")
        for u, v in graph_copy.edges():
            # graph_copy[u][-1]['weight'] = 0
            # service_idx_u = u
            if u != -2:
                dual_u = dual_values["Service_" + str(u)]
                graph_copy[u][v]['weight'] = -(dual_u)
            # if v == -1:
            #     sink_edge_weights.append(graph_copy[u][v]['weight'])

        # print("edge weights to sink",sink_edge_weights)
        
        # Initialize dictionaries for cost, cumulative duration, and predecessor pointers.
        nodes = list(graph_copy.nodes())
        INF = float('inf')
        cost = {node: INF for node in nodes}
        duration = {node: INF for node in nodes}
        pred = {node: None for node in nodes}
        
        # The source node (-2) has cost 0 and duration 0.
        cost[-2] = 0
        duration[-2] = 0
        
        # Perform up to (|V| - 1) relaxations.
        for _ in range(len(nodes) - 1):
            updated = False
            # For each edge, try to relax.
            for u, v in graph_copy.edges():
                # If u is reachable...
                if cost[u] < INF:
                    # Additional duration for node v: if v is a service node, add its service duration.
                    add_dur = service_dict[u].serv_dur if u not in [-2, -1] else 0
                    new_dur = duration[u] + add_dur
                    # Only relax if the new cumulative duration is within allowed limit.
                    if new_dur <= time_constr:
                        new_cost = cost[u] + graph_copy[u][v]['weight']
                        if new_cost < cost[v]:
                            cost[v] = new_cost
                            duration[v] = new_dur
                            pred[v] = u
                            updated = True
            # No updates in this iteration means we can stop early.
            if not updated:
                break

        # If the sink (-1) is unreachable within the duration constraint, return None.
        if cost[-1] == INF:
            return None, INF
        # Reconstruct the path from sink (-1) back to source (-2) using predecessor pointers.
        path = []
        current = -1
        while current is not None:
            path.append(current)
            current = pred[current]
        path.reverse()
        # Remove the source (-2) and sink (-1) from the reconstructed path.
        if path and path[0] == -2:
            path = path[1:]
        if path and path[-1] == -1:
            path = path[:-1]
        cost_final = -cost[-1]
        return path, cost_final
    # ////////////////////////////////////////////////////////////
    elif method == "label-setting":
            print("label-setting")
        # def dominates(lab1, lab2):
        #     """
        #     Returns True if lab1 dominates lab2.
        #     Each label is of the form: (cost, resource_vector, node, pred).
        #     lab1 dominates lab2 if:
        #     - lab1.cost <= lab2.cost, and
        #     - For every index i, lab1.resource_vector[i] <= lab2.resource_vector[i],
        #         with at least one strict inequality.
        #     (A safeguard converts an int resource into a one-element tuple.)
        #     """
        #     cost1, res1, node1, _ = lab1
        #     cost2, res2, node2, _ = lab2
        #     if not isinstance(res1, tuple):
        #         res1 = (res1,)
        #     if not isinstance(res2, tuple):
        #         res2 = (res2,)
        #     if cost1 > cost2:
        #         return False
        #     all_le = all(res1[i] <= res2[i] for i in range(len(res1)))
        #     any_lt = any(res1[i] < res2[i] for i in range(len(res1)))
        #     return all_le
            # return all_le and any_lt

    # def new_duty_rcsp_tent_perm(graph, dual_values, service_dict, max_resource):
            """
            RCSP algorithm with multiple vectorial labeling using explicit tentative (L) vs.
            permanent (P) label management and iterative refinement, as described in Aneja et al. (1983).
            
            Each label is represented as:
                (cost, resource_vector, node, pred)
            where resource_vector is a tuple (currently (duration,)) and pred is a pointer to the predecessor label.
            
            The algorithm:
            1. Initializes L(source) with (0, (0,), source, None) and sets P(source) = {}.
            2. Uses a priority queue (heap) to extract the best tentative label (lexicographic order: cost then resource_vector).
            3. When a label is popped from the heap, if it is still in L(u) it is finalized (moved to P(u)).
            4. The permanent label is then extended to each successor v, generating new labels that are added to L(v)
                only if they are not dominated (by either tentative or permanent labels already at v).
            5. An iterative refinement step is then performed on all nodes: for each node, any label in L(node) that is dominated
                by any label in P(node) is removed.
            6. The process continues until the heap is empty.
            7. At termination, the best label at the sink (node -1) is selected and its path is reconstructed.
            
            Parameters:
            graph         - A NetworkX DiGraph (with designated source = -2 and sink = -1).
            dual_values   - Dictionary mapping "service_{u}" to dual values.
            service_dict  - Dictionary mapping service numbers to Service objects (with attributes start_time, end_time, serv_dur).
            max_resource  - Maximum allowed resource (e.g. maximum duty duration) as a number.
            
            Returns:
            best_path      - The best path (list of nodes) from source (-2) to sink (-1) if one exists.
            best_cost      - The cost of that path.
            tentative      - Dictionary of tentative labels per node.
            permanent      - Dictionary of permanent labels per node.
            """
            source = -2
            sink = -1
            # //////////////////////////////  extrra check for prunning//////////////////////////////
            
            # Enforce tail charging resource values on edges using the "weight" attribute.
            # For an edge (x, y):
            # - If x == source, set weight = 0.
            # - If y == sink, set weight = service_dict[x].serv_dur (if available).
            # - Otherwise, set weight = service_dict[x].serv_dur.
            for x, y in list(graph.edges()):
                if x == source:
                    graph[x][y]['weight'] = 0
                elif y == sink:
                    try:
                        graph[x][y]['weight'] = service_dict[x].serv_dur
                    except KeyError:
                        graph[x][y]['weight'] = 0
                else:
                    try:
                        graph[x][y]['weight'] = service_dict[x].serv_dur
                    except KeyError:
                        graph[x][y]['weight'] = 0
            
            # Precompute the lower bound on resource from any node to the sink.
            # Here we use the "weight" attribute for the reversed graph.
            try:
                rev_graph = graph.reverse(copy=True)
                g_comp_res = nx.single_source_dijkstra_path_length(rev_graph, sink, weight="weight")
            except Exception as e:
                print("Error computing g_comp_res:", e)
                g_comp_res = {}
            # Initialize tentative set at source: store resource as tuple (0,)
            init_label = (0, (0,), source, None)
            tentative = {source: [init_label]}
            permanent = {source: []}
            
            # Priority queue (heap) contains tentative labels.
            heap = [init_label]
            
            while heap:
                # Pop the best tentative label.
                label = heapq.heappop(heap)  # label: (cost, res_vec, u, pred)
                cost, res_vec, u, pred = label
                # If this label is no longer in L(u), skip it.
                if u not in tentative or label not in tentative[u]:
                    continue
                
                # Finalize the label: remove it from tentative and add to permanent.
                tentative[u].remove(label)
                permanent.setdefault(u, []).append(label)
                
                # Extend this finalized label to every successor v.
                for v in graph.successors(u):
                    # Compute transition time if both u and v are real service nodes.
                    # trans_time = 0
                    # if u not in [source, sink] and v not in [source, sink]:
                    #     trans_time = max(0, service_dict[v].start_time - service_dict[u].end_time)
                    # Compute additional resource: service duration at v.
                    # add_dur = service_dict[v].serv_dur if v not in [source, sink] else 0
                    add_dur = service_dict[u].serv_dur if u not in [source, sink] else 0
                    new_duration = res_vec[0] +add_dur
                    # new_duration = res_vec[0] + trans_time + add_dur
                    # Look-ahead: check that even with an optimistic lower bound from v, total resource doesn't exceed max_resource.
                    if v in g_comp_res and new_duration + g_comp_res[v] > time_constr:
                        continue

                    if new_duration > time_constr:
                        continue  # Skip extension if resource constraint is violated.
                    new_res_vec = (new_duration,)
                    
                    # Update cost: subtract dual value for u if applicable.
                    if u != source:
                        # dual_value = dual_values.get(f"service_{u}", 0)
                        dual_u = dual_values["Service_" + str(u)]

                        add_cost = -(dual_u)
                    else:
                        add_cost= 0
                    new_cost = cost + add_cost
                    
                    # Create new label with predecessor pointer.
                    new_label = (new_cost, new_res_vec, v, label)
                    
                    # Dominance check at node v against tentative labels.
                    dominated = False
                    non_dominated = []
                    for lab in tentative.get(v, []):
                        if (lab[0] <= new_cost and lab[1] <= new_res_vec):
                            dominated = True
                            break
                        if not ((new_cost <= lab[0] and new_res_vec < lab[1])):
                            non_dominated.append(lab)
                    if dominated:
                        continue
                    # Also check against permanent labels at v.
                    for lab in permanent.get(v, []):
                        if (lab[0] < new_cost) or (lab[0] == new_cost and lab[1] <= new_res_vec):
                            dominated = True
                            break
                    if dominated:
                        continue
                    
                    # If not dominated, update tentative set for v.
                    tentative.setdefault(v, [])
                    tentative[v] = non_dominated + [new_label]
                    heapq.heappush(heap, new_label)
                
                # # Iterative Refinement: for every node, remove from tentative any label dominated by a permanent label.
                # for node in list(tentative.keys()):
                #     refined = []
                #     for lab in tentative[node]:
                #         if not any(dominates(perm_lab, lab) for perm_lab in permanent.get(node, [])):
                #             refined.append(lab)
                #     tentative[node] = refined
            
            # End loop: Gather all labels at sink (both tentative and permanent).
            sink_labels = tentative.get(sink, []) + permanent.get(sink, [])
            if sink_labels:
                best_label = min(sink_labels, key=lambda lab: (lab[0], lab[1]))
                best_path = reconstruct_path(best_label)
                # The accumulated dual sum is -best_label[0].
                best_cost = -(best_label[0]) 
                # best_cost = best_label[0]
                # return best_path, best_cost, tentative, permanent
                # Remove the source (-2) and sink (-1) from the reconstructed path.
                if best_path and best_path[0] == -2:
                    best_path = best_path[1:]
                if best_path and best_path[-1] == -1:
                    best_path = best_path[:-1]
                return best_path, best_cost
            else:
                return None, None   
#         import heapq

# # def new_duty_with_RCSP_priority_pred(graph, dual_values, service_dict, max_resource):
#         """
#         Finds a new duty (path from source -2 to sink -1) using a Resource-Constrained
#         Shortest Path (RCSP) algorithm with a priority queue and predecessor pointers.
        
#         Instead of storing the full path in each label, each label is stored as a tuple:
#         (cost, resource, node, pred)
#         where 'pred' is a pointer to the predecessor label. This is more memory efficient,
#         and the full path can be reconstructed later via backtracking.
        
#         The label tuple is ordered as (cost, resource, node, pred) so that if two labels have 
#         the same cost, the one with the lower resource consumption is processed first.
        
#         Parameters:
#         graph         - A NetworkX DiGraph.
#         dual_values   - Dictionary of dual values (e.g., {"service_1": value, ...}).
#         service_dict  - Dictionary mapping service numbers to Service objects.
#         max_resource  - Maximum allowed resource (e.g., maximum duty duration in minutes).
        
#         Returns:
#         best_path     - The best path (list of nodes) from source (-2) to sink (-1).
#         best_cost     - The associated cost (reduced cost) of that path.
#         labels        - Dictionary of labels at each node (for debugging).
#         """
#         # Here, we denote source as -2 and sink as -1.
#         # Label tuple: (cost, resource, node, pred)
#         # For the source, pred is None.
#         labels = { -2: [(0, 0, -2, None)] }
#         # The heap stores labels. It is ordered lexicographically, so (cost, resource, ...) works as desired.
#         heap = [(0, 0, -2, None)]
#         while heap:
#             cost, resource, u, pred = heapq.heappop(heap)
#             # Extend the label from node u to every successor v.
#             for v in graph.successors(u):
#                 # # Compute transition time only if both u and v are service nodes (not source or sink)
#                 # transition_time = 0
#                 # if u not in [-2, -1] and v not in [-2, -1]:
#                 #     transition_time = max(0, service_dict[v].start_time - service_dict[u].end_time)
                
#                 # Additional resource consumption is the service duration at v (if v is a service)
#                 additional_duration = service_dict[v].serv_dur if v not in [-2, -1] else 0
#                 # new_resource = resource + transition_time + additional_duration
#                 new_resource = resource + additional_duration
#                 if new_resource > time_constr:
#                     continue  # Skip if resource limit is exceeded
                
#                 # Update cost: subtract the dual value for u if u is not the source.
#                 # if u != -2:
#                 # service_idx_u = u
#                 # # dual_u = dual_values[service_idx_u]
#                 if u != -2:
#                     dual_u = dual_values["Service_" + str(u)]
#                 else:
#                     dual_u = 0
#                 additional_cost = -(dual_u)
#                 new_cost = cost + additional_cost
                
#                 # Create a new label. Instead of storing the full path, store a pointer (predecessor) to the current label.
#                 # We set current_label to the label we just popped.
#                 current_label = (cost, resource, u, pred)
#                 new_label = (new_cost, new_resource, v, current_label)
                
#                 # Dominance check at node v.
#                 dominated = False
#                 non_dominated = []
#                 for existing in labels.get(v, []):
#                     # Compare by cost and resource.
#                     if existing[0] <= new_cost and existing[1] <= new_resource:
#                         dominated = True
#                         break
#                     if not (new_cost <= existing[0] and new_resource <= existing[1]):
#                         non_dominated.append(existing)
#                 if dominated:
#                     continue
                
#                 # Update labels at node v.
#                 labels.setdefault(v, [])
#                 labels[v] = non_dominated + [new_label]
                
#                 # Push the new label onto the heap.
#                 heapq.heappush(heap, new_label)
        
#         # Termination: Check if any labels reached the sink (-1)
#         if -1 in labels and labels[-1]:
#             best_label = min(labels[-1], key=lambda x: x[0])
#             best_path = reconstruct_path(best_label)
#             best_cost = -(best_label[0])
#             # Remove the source (-2) and sink (-1) from the reconstructed path.
#             if best_path and best_path[0] == -2:
#                 best_path = best_path[1:]
#             if best_path and best_path[-1] == -1:
#                 best_path = best_path[:-1]
#             return best_path, best_cost
#         else:
#             return None, None

        

    elif method == "bellman ford":
        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 1
            else:
                # graph[u][v]['weight'] = -dual_values[u]
                graph[u][v]['weight'] = -dual_values["Service_" + str(u)]

        shortest_path = nx.shortest_path(graph, source=-2, target=-1, weight='weight', method = 'bellman-ford')
        shortest_distance = nx.shortest_path_length(graph, source=-2, target=-1, weight='weight', method = 'bellman-ford')

        if verbose:
            path_duals = []
            for i in range(len(shortest_path)-1):
                path_duals.append(graph[shortest_path[i]][shortest_path[i+1]]['weight'])

            print("Path Duals: ", path_duals)

        return shortest_path[1:-1], shortest_distance
                        

    elif method == "dp":

        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 0
            else:
                # graph[u][v]['weight'] = dual_values[u]
                graph[u][v]['weight'] = dual_values["Service_" + str(u)]

        dp_dict = defaultdict(dict)

        topo_order = list(nx.topological_sort(graph))
        # print(topo_order)
        # for time in range(time_constr):
        #     dp_dict[-2][time] = (0, None)

        for node in topo_order:
            for time in range(time_constr+1):
                if node == -2:
                    dp_dict[node][time] = (0, None)
                else:
                    best_pred = None
                    best = 0
                    for pred in graph.predecessors(node):
                        # if graph.nodes[pred]["service_time"] > time:
                        if pred in [-1,-2] :pred_dur = 0 
                        else: pred_dur = service_dict[pred].serv_dur
                        if pred_dur > time:
                            continue
                        # time_range = max(0, time - graph.nodes[node]["service_time"])
                        # if node in [-1,-2] :node_dur = 0 
                        # else: node_dur = service_dict[node].serv_dur
                        time_range = max(0, time - pred_dur)
                        for t in range(time_range + 1):
                            current = dp_dict[pred][t][0] + graph[pred][node]['weight']
                            # print("Pred: ", pred, ", Time: ", time, ", T: ",t , "Current: ",current, "dp_dict", dp_dict[pred][t])
                            if current >= best:
                                best = current
                                best_pred = pred
                    dp_dict[node][time] = (best, best_pred)
        # print(dp_dict[-1])


        #extracting path
        remaining_time = time_constr
        current = -1
        spprc = dp_dict[current][remaining_time][0]

        path = deque()

        while current != -2:
            path.appendleft(current)
            # print(current, remaining_time)
            pred = dp_dict[current][remaining_time][1]
            # remaining_time -= graph.nodes[current]["service_time"]
            if pred in [-1,-2] :pred_dur = 0 
            else: pred_dur = service_dict[pred].serv_dur
            
            remaining_time -= pred_dur
            # remaining_time = max(0, remaining_time)
            current =pred
            # print("Shortest Path: ", path)  

        print("Shortest Path Length: ", spprc)
        print("Shortest Path: ", path)

        return list(path)[:-1], spprc

    elif method == "ip":
        service_dict[-1] = Service([-1,-1, "x", "00:00", "x", "00:00","x",0, "x", "x"])
        service_dict[-2] = Service([-2,-2, "x", "00:00", "x", "00:00","x",0, "x", "x"])

        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 0
            else:
                # graph[u][v]['weight'] = dual_values[u]
                graph[u][v]['weight'] = dual_values["Service_" + str(u)]

        model = gp.Model("SPPRC")
        model.setParam('OutputFlag', 0) 

        incoming_var = defaultdict(list)
        outgoing_var = defaultdict(list) 
        edge_vars = {} #xij - binary

        incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

        #Decision Variables
        for i,j in graph.edges():
            edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            incoming_var[j].append(edge_vars[i,j])
            outgoing_var[i].append(edge_vars[i,j])
            # print(edge_vars)
            # print("Edge: ", i, " ", j)
            # model.update()
            # print(edge_vars[(i,j)].VarName)

        #Objective 
        model.setObjective(gp.quicksum((edge_vars[i,j]*graph[i][j]['weight']) for i,j in graph.edges()), GRB.MAXIMIZE)


        #Constraints - Flow conservation
        flow_constraints = []
        for i in graph.nodes():
            if i == -2:
                constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"Service_flow_{i}") 
                flow_constraints.append(constr)
            elif i == -1:
                constr = model.addConstr(gp.quicksum(incoming_var[i]) == 1,name=f"Service_flow_{i}") 
                flow_constraints.append(constr)
            else:
                constr = model.addConstr(gp.quicksum(incoming_var[i])== gp.quicksum(outgoing_var[i]),name=f"Service_inflow_{i}") 
                # constr2 = model.addConstr(gp.quicksum(outgoing_var[i]) ==1, name=f"Service_outflow_{i}")

                flow_constraints.append(constr)
                # flow_constraints.append(constr2)


        
        #resource constraint 
        # duration = sum([(edge_vars[i,j].x *service_dict[i].serv_dur) for i,j in graph.edges() if i not in [-1,-2]])
        # print("Total Duration: ", duration)
        model.addConstr(gp.quicksum((edge_vars[i,j]*service_dict[i].serv_dur) for i,j in graph.edges() if i !=-1) <= time_constr)
        

        # model.update()
        model.write("model_dp.lp")
        model.optimize()

        if model.status == GRB.INFEASIBLE:
            print ("Hehe!")
            print("Model is not feasible")
        #extracting path

        reduced_cost = 0
        path = []
        current = -2
        path.append(current) 
        while current != -1:
            for var in outgoing_var[current]:
                if var.x ==1:
                    _ , next = extract_nodes(var.VarName)
                    reduced_cost += graph[current][next]['weight']
                    path.append(next)
                    current = next
                    break

        del service_dict[-1]
        del service_dict[-2]

        return path[1:-1], reduced_cost
        
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
# def generate_new_column_3(graph, service_dict, dual_values, method = "dp", verbose = False, time_constr = 6*60):



def mip(service_dict, duties, show_solutions = True, show_objective = True,warm =140):


    model = gp.Model("CrewScheduling")
    # model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 600)

    
    ###Decision Variables
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.BINARY, name=f"x{i}"))

    big_penalty = 1e6


    ### Objective
    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    
    ### Constraints
    service_constraints = []
    for service_idx, service in enumerate(service_dict.values()):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty)>= 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)


    ##warm start
    for i in range(warm):
        duty_vars[i].Start = 1


    model.optimize()



    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
        return None, None, None, None
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        # model.write("model.lp") 
        if show_solutions:
            print("Optimal solution found")
        
        # Get the dual variables for each service constraint
        # dual_values = [constr.Pi for constr in service_constraints]
        # duals = dict([(constr.ConstrName, constr.Pi) for constr in service_constraints])

        
        solution = [v.x for v in model.getVars()]
        # print("Hi" ,len(solution))
        selected_duties = [(v.varName, v.x) for v in model.getVars() if v.x > 0]
        selected_duties_vars = [v for v in model.getVars() if v.x > 0]

        if show_solutions:
            print("Positive Duties, 0: ", len(selected_duties))
            for variable in selected_duties_vars:
                print(variable.varName, variable.x)
        if show_objective:    
            print(f"Objective Value: {objective.getValue()}")

        return objective.getValue(), selected_duties
    else:
        print("No optimal solution found.")
        return None, None