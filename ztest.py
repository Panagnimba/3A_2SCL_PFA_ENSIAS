import random
import numpy as np

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

    return best_solution, best_fitness, fitness_history


# Example of how to test the algorithm
if _name_ == "_main_":
    num_levels = 3
    num_facilities = 5
    num_clients = 10
    num_cap_levels = 3
    pop_size = 10
    num_generations = 50

    costs, capacities, demand, budget, opening_costs = generate_problem_data(
        num_levels, num_facilities, num_clients, num_cap_levels
    )

    best_solution, best_fitness, fitness_history = genetic_algorithm(
        pop_size, num_generations, costs, capacities, demand, budget, opening_costs
    )

    print("Best solution found:", best_solution)
    print("Best fitness value:", best_fitness)
    print("Fitness history:", fitness_history)