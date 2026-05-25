import random
import itertools
import time
import matplotlib.pyplot as plt

N_FIELDS = 6
K_CROPS = 3

yield_matrix = [
    [10, 15, 8],
    [12, 10, 20],
    [8, 20, 15],
    [14, 12, 10],
    [10, 10, 10],
    [25, 5, 15]
]

crop_costs = [5, 10, 7]

def calculate_fitness(chromosome):
    total_yield = 0
    total_cost = 0
    for field_idx, crop_idx in enumerate(chromosome):
        total_yield += yield_matrix[field_idx][crop_idx]
        total_cost += crop_costs[crop_idx]
    return total_yield / total_cost

def brute_force():
    best_score = -1
    best_solution = None
    all_combinations = itertools.product(range(K_CROPS), repeat=N_FIELDS)
    for combo in all_combinations:
        score = calculate_fitness(combo)
        if score > best_score:
            best_score = score
            best_solution = combo
    return best_solution, best_score

def tournament_selection(pop):
    candidates = random.sample(pop, 3)
    return max(candidates, key=lambda x: calculate_fitness(x))

def uniform_crossover(parent1, parent2):
    child = []
    for i in range(N_FIELDS):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def multipoint_crossover(parent1, parent2):
    points = sorted(random.sample(range(1, N_FIELDS), 2))
    child = []
    use_parent1 = True
    start = 0
    for point in points:
        if use_parent1:
            child.extend(parent1[start:point])
        else:
            child.extend(parent2[start:point])
        use_parent1 = not use_parent1
        start = point
    if use_parent1:
        child.extend(parent1[start:])
    else:
        child.extend(parent2[start:])
    return child

def genetic_algorithm(crossover_func, pop_size=20, generations=50, mutation_rate=0.1):
    population = [[random.randint(0, K_CROPS - 1) for _ in range(N_FIELDS)] for _ in range(pop_size)]
    best_fitness_history = []

    for _ in range(generations):
        population = sorted(population, key=lambda x: calculate_fitness(x), reverse=True)
        best_fitness_history.append(calculate_fitness(population[0]))

        next_gen = [population[0], population[1]]

        while len(next_gen) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover_func(parent1, parent2)


            if random.random() < mutation_rate:
                idx = random.randint(0, N_FIELDS - 1)
                child[idx] = random.randint(0, K_CROPS - 1)
                id1, id2 = random.sample(range(N_FIELDS), 2)
                child[id1], child[id2] = child[id2], child[id1]
                start, end = sorted(random.sample(range(N_FIELDS), 2))
                child[start:end+1] = reversed(child[start:end+1])

            next_gen.append(child)

        population = next_gen

    best_solution = max(population, key=lambda x: calculate_fitness(x))
    return best_solution, calculate_fitness(best_solution), best_fitness_history


start_bf = time.time()
bf_sol, bf_score = brute_force()
bf_time = time.time() - start_bf
print(f"Полный перебор: {bf_sol}, Эффективность: {bf_score:.4f}, Время: {bf_time:.4f} с")


start_u = time.time()
ga_sol_u, ga_score_u, ga_hist_u = genetic_algorithm(uniform_crossover)
ga_time_u = time.time() - start_u
print(f"ГА (uniform): {ga_sol_u}, Эффективность: {ga_score_u:.4f}, Время: {ga_time_u:.4f} с")


start_m = time.time()
ga_sol_m, ga_score_m, ga_hist_m = genetic_algorithm(multipoint_crossover)
ga_time_m = time.time() - start_m
print(f"ГА (multi-point): {ga_sol_m}, Эффективность: {ga_score_m:.4f}, Время: {ga_time_m:.4f} с")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))


axes[0].plot(ga_hist_u, label=f'Uniform (фитнес {ga_score_u:.4f})', color='blue', marker='.')
axes[0].plot(ga_hist_m, label=f'Multi-point (фитнес {ga_score_m:.4f})', color='green', marker='.')
axes[0].axhline(y=bf_score, color='red', linestyle='--', label=f'Оптимум ({bf_score:.4f})')
axes[0].set_title('Сходимость ГА с разными скрещиваниями')
axes[0].set_xlabel('Поколение')
axes[0].set_ylabel('Лучшая приспособленность')
axes[0].legend()
axes[0].grid(True)


labels = ['Uniform', 'Multi-point']
scores = [ga_score_u, ga_score_m]
times = [ga_time_u, ga_time_m]
x = range(len(labels))
width = 0.35

bars1 = axes[1].bar(x, scores, width, label='Эффективность', color='steelblue')
axes[1].set_ylabel('Эффективность (баллы)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)

ax2 = axes[1].twinx()
bars2 = ax2.bar([p + width for p in x], times, width, label='Время (с)', color='orange')
ax2.set_ylabel('Время (сек)')


handles = [bars1, bars2]
legend_labels = ['Эффективность', 'Время']
axes[1].legend(handles, legend_labels, loc='upper left')
axes[1].set_title('Сравнение методов скрещивания')

plt.tight_layout()
plt.show()
