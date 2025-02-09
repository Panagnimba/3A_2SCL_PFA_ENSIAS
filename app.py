from flask import Flask, render_template
from docplex.mp.model import Model
import numpy as np
import random
from flask import Flask, render_template, request,jsonify
import csv
import os
import json

app = Flask(__name__)
app.debug = True  # Enables auto-reload

@app.route('/')
def home():
    return render_template('index.html', show_input_form=True)

@app.route('/cplex_solver')
def cplex_solver():
    return render_template('cplex_solver.html', show_input_form=True)

@app.route('/genetic_solver')
def genetic_solver():
    return render_template('genetic_solver.html', show_input_form=True)





from cplex import Cplex
from cplex.exceptions import CplexError
from cplex import SparsePair
@app.route('/solve_cplex_algorithm', methods=['POST'])
def solve_genetic_algorithm():

    # Problem Parameters
    N = 5  # Number of clients
    nfact = 3  # Number of factories
    nwh = 4  # Number of warehouses
    K = 3  # Capacity levels for warehouses
    L = 5  # Assignment levels

    clients = range(1, N + 1)
    warehouses = range(1, nwh + 1)
    levels = range(1, L + 1)
    cap_levels = range(1, K + 1)

    # Data
    d = {i: 10 for i in clients}  # Demand per client
    u_wh = {(i, k): 20 for i in warehouses for k in cap_levels}  # Warehouse capacity
    f_wh = {(i, k): 50 for i in warehouses for k in cap_levels}  # Warehouse opening costs
    c = {(i, j, l): 5 for i in warehouses for j in clients for l in levels}  # Cost warehouse-client
    B = 5000  # Total budget

    # Initialize CPLEX problem
    problem = Cplex()
    problem.set_problem_type(Cplex.problem_type.LP)

    # Decision Variables
    x = {}
    y = {}

    # Add variables for warehouse-client assignment
    for i in warehouses:
        for j in clients:
            for l in levels:
                var_name = f"x_{i}_{j}_{l}"
                x[(i, j, l)] = var_name
                problem.variables.add(names=[var_name], types="C", lb=[0])

    # Add variables for warehouse opening
    for i in warehouses:
        for k in cap_levels:
            for l in levels:
                var_name = f"y_{i}_{k}_{l}"
                y[(i, k, l)] = var_name
                problem.variables.add(names=[var_name], types="B", lb=[0], ub=[1])

    # Objective Function: Minimize total cost
    objective = []
    for i in warehouses:
        for j in clients:
            for l in levels:
                objective.append((x[(i, j, l)], c[(i, j, l)]))
        for k in cap_levels:
            for l in levels:
                objective.append((y[(i, k, l)], f_wh[(i, k)]))

    problem.objective.set_linear(objective)
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Constraints

    # Constraint 1: Each client assigned to at least one warehouse at each level
    for j in clients:
        for l in levels:
            constraint = [(x[(i, j, l)], 1) for i in warehouses]
            problem.linear_constraints.add(
                lin_expr=[SparsePair(*zip(*constraint))],
                senses="E",
                rhs=[d[j]],
            )

    # Constraint 2: Warehouse capacity at each level
    for i in warehouses:
        for l in levels:
            constraint = [(x[(i, j, l)], 1) for j in clients]
            for k in cap_levels:
                constraint.append((y[(i, k, l)], -u_wh[(i, k)]))
            problem.linear_constraints.add(
                lin_expr=[SparsePair(*zip(*constraint))],
                senses="L",
                rhs=[0],
            )

    # Constraint 3: Limit one capacity level per warehouse per level
    for i in warehouses:
        for l in levels:
            constraint = [(y[(i, k, l)], 1) for k in cap_levels]
            problem.linear_constraints.add(
                lin_expr=[SparsePair(*zip(*constraint))],
                senses="L",
                rhs=[1],
            )

    # Constraint 4: Total budget
    constraint = []
    for i in warehouses:
        for k in cap_levels:
            for l in levels:
                constraint.append((y[(i, k, l)], f_wh[(i, k)]))
    problem.linear_constraints.add(
        lin_expr=[SparsePair(*zip(*constraint))],
        senses="L",
        rhs=[B],
    )

    # Constraint 5: Balance flow across levels
    for j in clients:
        for l in levels:
            if l > 1:
                constraint = [(x[(i, j, l)], 1) for i in warehouses]
                constraint += [(x[(i, j, l - 1)], -1) for i in warehouses]
                problem.linear_constraints.add(
                    lin_expr=[SparsePair(*zip(*constraint))],
                    senses="G",
                    rhs=[0],
                )

    # Solve the problem
    try:
        problem.solve()
        best_solution =  problem.solution.get_objective_value()
        solution_status = problem.solution.get_status_string()
        print("Solution status:", solution_status)
        print("Objective value:",best_solution)

        # Print variable values
        variable_values = problem.solution.get_values()
        variable_names = problem.variables.get_names()

        # print("\nVariable values:")
        # for name, value in zip(variable_names, variable_values):
        #     print(f"{name}: {value}")
         # Prepare response
        response = {
            "solution_status": solution_status,
            "objective_value": best_solution,
            "variable_values": {name: value for name, value in zip(variable_names, variable_values)},
        }
        print(response["variable_values"])
        return render_template('cplex_result.html', data=response)

    except CplexError as e:
        print("CPLEX Error:", e)

    

# Generate problem data
def generate_problem_data(num_levels, num_facilities, num_clients, num_cap_levels):
    costs = [
        {(i, j): random.randint(1, 20) for i in range(num_facilities) for j in range(num_clients)}
        for _ in range(num_levels)
    ]
    capacities = [
        np.random.randint(10, 50, size=(num_facilities, num_cap_levels))
        for _ in range(num_levels)
    ]
    demand = np.random.randint(5, 20, size=num_clients)
    budget = random.randint(100, 500)
    opening_costs = [
        np.random.randint(10, 30, size=(num_facilities, num_cap_levels))
        for _ in range(num_levels)
    ]
    return costs, capacities, demand, budget, opening_costs

# Evaluate fitness
def evaluate_fitness(solution, costs, capacities, demand, budget, opening_costs):
    y, x = solution
    total_cost = 0
    penalty = 0

    # Objective function: Minimize total cost
    for l, cost_level in enumerate(costs):
        for (i, j), c in cost_level.items():
            total_cost += c * max(0, x[l][i][j])

    opening_cost = 0
    for l in range(len(opening_costs)):
        for i in range(opening_costs[l].shape[0]):
            for k in range(opening_costs[l].shape[1]):
                opening_cost += y[l][i][k] * opening_costs[l][i][k]

    total_cost += opening_cost

    # Constraint 1: Each client is assigned to at least one installation per level
    for j in range(len(demand)):
        if not np.isclose(np.sum(x[0, :, j]), demand[j]):
            penalty += 50 * abs(np.sum(x[0, :, j]) - demand[j])

    # Constraint 2: Capacity constraints
    for l in range(len(capacities)):
        for i in range(capacities[l].shape[0]):
            if np.sum(x[l][i]) > np.sum(capacities[l][i] * y[l][i]):
                penalty += 100 * (np.sum(x[l][i]) - np.sum(capacities[l][i] * y[l][i]))

    # Constraint 3: Single capacity level per facility
    for l in range(len(capacities)):
        for i in range(capacities[l].shape[0]):
            if np.sum(y[l][i]) > 1:
                penalty += 100

    # Constraint 4: Budget constraint
    if opening_cost > budget:
        penalty += 100 * (opening_cost - budget)

    # Constraint 5: Flow balance across assignment levels
    for l in range(1, len(capacities)):
        for j in range(len(demand)):
            if np.sum(x[l, :, j]) < np.sum(x[l - 1, :, j]):
                penalty += 50 * abs(np.sum(x[l - 1, :, j]) - np.sum(x[l, :, j]))

    fitness = total_cost + penalty
    return fitness

# Repair solutions to ensure feasibility
def repair_solution(solution, capacities, demand):
    y, x = solution
    for l in range(len(capacities)):
        for i in range(capacities[l].shape[0]):
            if np.sum(x[l][i]) > np.sum(capacities[l][i] * y[l][i]):
                x[l][i] = np.sum(capacities[l][i] * y[l][i]) * (x[l][i] / np.sum(x[l][i]))
    return y, x

# Genetic algorithm
# Array to store the best fitness at each generation
 
def genetic_algorithm(pop_size, num_generations, costs, capacities, demand, budget, opening_costs):
    fitness_history = [] 
    num_levels = len(capacities)
    num_facilities = capacities[0].shape[0]
    num_clients = len(demand)
    num_cap_levels = capacities[0].shape[1]

    def initialize_population():
        population = []
        for _ in range(pop_size):
            y = np.random.randint(2, size=(num_levels, num_facilities, num_cap_levels))
            x = np.random.rand(num_levels, num_facilities, num_clients)
            population.append((y, x))
        return population

    def tournament_selection(population, fitness_scores, k=3):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(fitness_scores, population)), k)
            selected.append(min(tournament, key=lambda x: x[0])[1])
        return selected

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, num_levels - 1)
        y_child = np.vstack((parent1[0][:crossover_point], parent2[0][crossover_point:]))
        x_child = np.vstack((parent1[1][:crossover_point], parent2[1][crossover_point:]))
        return (y_child, x_child)

    def mutate(solution, mutation_rate=0.1):
        y, x = solution
        for l in range(y.shape[0]):
            for i in range(y.shape[1]):
                if random.random() < mutation_rate:
                    y[l][i] = 1 - y[l][i]
        return y, x

    population = initialize_population()
    best_solution = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        fitness_scores = [
            evaluate_fitness(sol, costs, capacities, demand, budget, opening_costs)
            for sol in population
        ]
        ranked_population = sorted(zip(fitness_scores, population), key=lambda x: x[0])
        population = tournament_selection([sol for _, sol in ranked_population], fitness_scores)

        if ranked_population[0][0] < best_fitness:
            best_fitness = ranked_population[0][0]
            best_solution = ranked_population[0][1]

        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            child = repair_solution(child, capacities, demand)
            new_population.append(child)

        population.extend(new_population)
        print(f"Generation {generation}, Best fitness: {best_fitness}")
        fitness_history.append((generation, best_fitness))  # Append to the array



        

    return best_solution, best_fitness,fitness_history

# Run the algorithm
# num_levels = 5
# num_facilities = 10
# num_clients = 15
# num_cap_levels = 3
# pop_size = 20
# num_generations = 50

# costs, capacities, demand, budget, opening_costs = generate_problem_data(
#     num_levels, num_facilities, num_clients, num_cap_levels
# )

# best_solution, best_fitness = genetic_algorithm(pop_size, num_generations, costs, capacities, demand, budget, opening_costs)
# print("Best solution found:", best_solution)
# print("Best fitness value:", best_fitness)


@app.route('/solve_genetic_algorithm', methods=['POST'])
def solve_genetic():
    try:
       
       # Extract form data
        num_levels = int(request.form['num_levels'])
        num_facilities = int(request.form['num_facilities'])
        num_clients = int(request.form['num_clients'])
        num_cap_levels = int(request.form['num_cap_levels'])
        pop_size = int(request.form['pop_size'])
        num_generations = int(request.form['num_generations']) 

        # Process CPLEX solver logic (not implemented here)
        costs, capacities, demand, budget, opening_costs = generate_problem_data(
                num_levels, num_facilities, num_clients, num_cap_levels
            )

        best_solution, best_fitness,fitness_history = genetic_algorithm(pop_size, num_generations, costs, capacities, demand, budget, opening_costs)
        print("Best solution found:", best_solution)
        # print("fitness array", fitness_history)
        print("Best fitness value:", best_fitness)

         # Return the results as JSON
        return render_template('result.html', best_solution=best_solution, best_fitness=best_fitness,fitness_history=fitness_history)

    except Exception as e:
        return render_template('index.html', error_message=f"Error: {str(e)}")







if __name__ == '__main__':
    app.run(debug=True)