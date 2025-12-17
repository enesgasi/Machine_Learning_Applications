import numpy as np

equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
num_weights = 6
sol_per_population = 8
num_parents_mating = 4
pop_size = (sol_per_population, num_weights)

new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
print("Initial population:\n", new_population)

def calc_pop_fitness(equation_inputs, pop):
    # calculate the fitness value of each solution
    fitness = np.sum(np.array(equation_inputs) * pop, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # create parents array with correct shape
    parents = np.empty((num_parents, pop.shape[1]))
    fitness_copy = fitness.copy()  # don't modify original fitness

    for parent_num in range(num_parents):
        max_idx = np.argmax(fitness_copy)            # index of best fitness
        parents[parent_num, :] = pop[max_idx, :]
        fitness_copy[max_idx] = -np.inf              # mark used
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = offspring_size[1] // 2
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring):
    # mutate a single gene (index 4) for each offspring
    for idx in range(offspring.shape[0]):
        random_value = float(np.random.uniform(-1.0, 1.0))
        offspring[idx, 4] = offspring[idx, 4] + random_value
    return offspring

num_generations = 5
for generation in range(num_generations):
    print("\nGeneration:", generation)
    fitness = calc_pop_fitness(equation_inputs, new_population)

    # select parents
    parents = select_mating_pool(new_population, fitness, num_parents_mating)

    # create offspring by crossover
    offspring_size = (sol_per_population - parents.shape[0], num_weights)
    offspring_crossover = crossover(parents, offspring_size)

    # apply mutation to offspring
    offspring_mutation = mutation(offspring_crossover)

    # create new population (elitism: keep parents)
    new_population[0: parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # print best fitness (compute along axis 1)
    best_val = np.max(np.sum(new_population * np.array(equation_inputs), axis=1))
    print("Best result:", best_val)

# final best solution and fitness
fitness = calc_pop_fitness(equation_inputs, new_population)
best_idx = int(np.argmax(fitness))
print("\nBest solution (weights):", new_population[best_idx, :])
print("Best solution fitness:", fitness[best_idx])
