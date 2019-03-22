import random

# one parent selection function
def select_parent(population, available_ids, tournament_size, first_parent_id):

    tournament = []
    # drawing parents from a population without replacement
    for i in range(0, int(tournament_size/100*len(population))):
        candidate = random.randrange(0, len(available_ids))
        while (available_ids[candidate] in tournament) or (available_ids[candidate] == first_parent_id):
            candidate = random.randrange(0, len(available_ids))
        else:
            tournament.append(available_ids[candidate])

    # choosing the best parent for crossover
    best_candidate = tournament[0]
    for candidate in tournament:
        if population[candidate].bacc > population[best_candidate].bacc:
            best_candidate = candidate

    parent = best_candidate

    return parent


# tournament selection of parents for crossover
def select(population, available_ids, tournament_size):

    first_parent_id = select_parent(population, available_ids, tournament_size, -1)
    second_parent_id = select_parent(population, available_ids, tournament_size, first_parent_id)

    return first_parent_id, second_parent_id