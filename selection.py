import random


# one parent selection function
def select_parent(population, tournament_size, first_parent_id):

    #create empty tournament
    tournament = []

    # drawing parents from a population without replacement
    for i in range(0, int(tournament_size*len(population))):
        candidate = random.randrange(0, len(population))  # randomly choose a candidate
        while population[candidate] == first_parent_id:  # check if ids of parent are unique
            candidate = random.randrange(0, len(population))
        else:
            tournament.append(candidate)  # add candidate to the tournament

    # choosing the best parent for crossover
    best_candidate = tournament[0]  # assign first in tournament as the best candidate
    for candidate in tournament:  # compare the best candidate to others
        if population[candidate].score > population[best_candidate].score:  # if new candidate is better
            best_candidate = candidate  # assign to the best candidate

    return best_candidate


# tournament selection of parents for crossover
def select(population, tournament_size):

    # select first parent (parents must have different ids!)
    first_parent_id = select_parent(population, tournament_size, -1)
    # select second parent (parents must have different ids!)
    second_parent_id = select_parent(population, tournament_size, first_parent_id)

    return first_parent_id, second_parent_id