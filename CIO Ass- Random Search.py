

import pandas as pd
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import random
import numpy as np

# Read the CSV file
df_ai= pd.read_csv('MScAI.csv')
df_ds= pd.read_csv('DSBA.csv')
df_se= pd.read_csv('MScSE.csv')


course_names = ['MSc in Artificial Intelligence', 
                'MSc in Data Science and Business Analytics', 
                'MSc in Software Engineering']

modules_ai = ['AI', 'NLP', 'IPCV', 'AML', 'CIO', 'FL','AR','PR','ESKE','BIS','MMDA','DL']
electives_ai = ['AR/PR','ESKE/DL','BIS/MMDA']

modules_ds = ['DM','MMDA','BDAT','BIS','DAP','AML','ABAV','BSSMMA', 'TSF','MDA','SEM','ORO','CIS', 'DL','NLP', 'BIA', 'DPM']
common_ds_modules = ['DM','MMDA','BDAT','BIS','DAP','AML','ABAV', 'MDA','SEM','ORO','NLP', 'BIA', 'DPM']
# Pathway-specific modules
pathway_modules_de = ['BSSMMA', 'TSF']
pathway_modules_elective_de = ['MDA','SEM','ORO']
pathway_modules_bi = ['CIS', 'DL']
pathway_modules_elective_bi = ['NLP', 'BIA', 'DPM']

modules_se = ['MSDP','RELM','OOSSE','SESE','SQE','SECT','IA','NDP','BDAT','DM','NLP']
electives_se = ['IA','NDP','BDAT','DM','NLP']

#first semester modules
first_sem_ai = ['AI','MMDA','IPCV','AML','FL']
first_sem_ds = ['AML','DM','BIS','DAP']
first_sem_se = ['DM','RELM','SECT']

# Preprocess the DataFrame to create a nested dictionary where the outer dictionary's keys
# are student names, the inner dictionary's keys are course names, and the values are lists of taken modules.

def read_data(data):
    students_data = {}
    for _, row in data.iterrows():
        name = row['NAME']
        intake_code = row['INTAKE CODE']
        course_name = row['COURSE NAME']
        module_code = row['MODULE CODE']
        
        # Split the module_code string at each hyphen and keep only the last part
        module_code = module_code.split('-')[-1]

        # Extract pathway from intake_code
        pathway = 'DE' if '(DE)' in intake_code else 'BI' if '(BI)' in intake_code else None

        if pathway is not None:
            course_name = f"{course_name}, {pathway}"

        if name not in students_data:
            students_data[name] = {course_name: {module_code}}
        else:
            if course_name in students_data[name]:
                students_data[name][course_name].add(module_code)
            else:
                students_data[name][course_name] = {module_code}

    return students_data


def create_individual():   
    # Separate chromosomes for each course
    chromosome_ai = create_course_chromosome(modules_ai)
    chromosome_ds = create_course_chromosome(modules_ds)
    chromosome_se = create_course_chromosome(modules_se)
    
    return [chromosome_ai, chromosome_ds, chromosome_se]

def create_course_chromosome(modules):
    # Initialize the chromosome with 5 intakes, RMCE in 1st, 3rd, and 5th intake
    chromosome = [['RMCE'], [], ['RMCE'], [], ['RMCE']]

    # For each intake in the chromosome
    for intake in chromosome:
        # Fill up the intake with modules, without repeating in two consecutive intakes
        num = random.randint(2,5)
        while len(intake) < num:  
            # Randomly select a module
            module = random.choice(modules)
            
            # Check that module is not in previous intake
            if module not in intake and (chromosome.index(intake) == 0 or module not in chromosome[chromosome.index(intake) - 1]):
                intake.append(module)
    
    return chromosome

def is_elective(module, electives):
    for elective_pair in electives:
        if module in elective_pair.split('/'):
            return True
    return False

def count_taken_electives(taken_modules, electives):
    count = 0
    for elective_pair in electives:
        elective_modules = elective_pair.split('/')
        if any(elective in taken_modules for elective in elective_modules):
            count += 1
    return count

def calc_fitness(chromosome, students_data, course_names):
    penalty_students_no_module = 0
    shared_modules_reward = 0
    first_semester_reward = 0
    penalty_consecutive_intakes = 0
    module_cost_penalty = 0
    module_inclusion_reward = 0
    cost = 0

    # Course-specific modules
    course_modules = {
        'MSc in Artificial Intelligence': modules_ai,
        'MSc in Data Science and Business Analytics': modules_ds,
        'MSc in Software Engineering': modules_se
    }

    # Course-specific electives
    course_electives = {
        'MSc in Artificial Intelligence': electives_ai,
        'MSc in Data Science and Business Analytics': None,  # To be defined based on pathway
        'MSc in Software Engineering': electives_se
    }
    
    shared_modules = {
        ('MSc in Artificial Intelligence', 'MSc in Data Science and Business Analytics'): 
            ['NLP', 'MMDA', 'BIS', 'DL', 'AML','ESKE'],
        ('MSc in Artificial Intelligence', 'MSc in Software Engineering'): ['NLP', 'DM'],
        ('MSc in Data Science and Business Analytics', 'MSc in Software Engineering'): 
            ['NLP', 'BDAT', 'DM'],
    }
    
    first_semester_modules = {
        'MSc in Artificial Intelligence': first_sem_ai,
        'MSc in Data Science and Business Analytics': first_sem_ds,
        'MSc in Software Engineering': first_sem_se
    }

    
    # Iterate over each intake
    for intake_index in range(len(chromosome[0])):
        # Set to store unique modules in the intake
        unique_modules = set()
        # Iterate over each course
        for course_chromosome in chromosome:
            # Add modules from this course in the intake to the set
            unique_modules.update(course_chromosome[intake_index])
        # Add the cost of unique modules to the total cost
        module_cost_penalty += len(unique_modules)
        total_cost = module_cost_penalty*5000
        
    # Iterate over students' data
    for student, data in students_data.items():
        for course, taken_modules in data.items():
            # Get the right course chromosome
            course_only = course.split(',')[0]
            course_index = course_names.index(course_only)
            course_chromosome = chromosome[course_index]

            # Pathway-specific modules (Data Science course only)
            if 'DE' in course:
                pathway_modules = pathway_modules_de
                pathway_electives = pathway_modules_elective_de
                course_electives['MSc in Data Science and Business Analytics'] = pathway_electives
            elif 'BI' in course:
                pathway_modules = pathway_modules_bi
                pathway_electives = pathway_modules_elective_bi
                course_electives['MSc in Data Science and Business Analytics'] = pathway_electives
                
            else:
                pathway_modules = None
            
            
            has_skipped = False  # Initialize the skip flag
            
            # Check for student penalty and pathway-specific modules reward
            for intake_index in range(len(course_chromosome)):
                intake = course_chromosome[intake_index]
                modules_to_take = [module for module in intake if module not in taken_modules]
                
                # Check for first semester reward
                if any(mod in first_semester_modules[course_only] for mod in intake):
                    first_semester_reward += 1
                
                if 'RMCE' in taken_modules and not modules_to_take and has_skipped == False:
                    has_skipped = True
                    continue
                
                if not modules_to_take:  # If no new module to take in this intake
                    if has_skipped:  # If the student has already skipped once, apply penalty
                        penalty_students_no_module += 1
                    elif 'RMCE' in taken_modules and has_skipped == False:  # Otherwise, allow the student to skip this intake without penalty
                        has_skipped = True
                else:  # If there are new modules to take
                    if course_only == 'MSc in Data Science and Business Analytics':
                        # Check if all modules to take are from a different pathway
                        if all(mod not in common_ds_modules and mod not in pathway_modules for mod in modules_to_take):
                            if 'RMCE' not in modules_to_take:
                                penalty_students_no_module += 1  # Penalty for taking a module not in the student's pathway   
                   
                    # Check for electives in modules to take and count taken electives
                    if course_only == 'MSc in Artificial Intelligence':
                        if all(is_elective(module, course_electives[course_only]) for module in modules_to_take) and \
                           count_taken_electives(taken_modules, course_electives[course_only]) >= 3:
                            if 'RMCE' in taken_modules and has_skipped == False:  # Otherwise, allow the student to skip this intake without penalty
                                has_skipped = True
                            else:
                                penalty_students_no_module += 1
                
                    # If all the non-taken modules are electives and the student has taken the maximum number of electives
                    else:
                        if all(module in course_electives[course_only] for module in modules_to_take) and \
                           len([module for module in taken_modules if module in course_electives[course_only]]) >= (3 if course_only != 'MSc in Data Science and Business Analytics' else 1):
                            if has_skipped:  # If the student has already skipped once, apply penalty
                                penalty_students_no_module += 1
                            elif 'RMCE' in taken_modules and has_skipped == False:  # Otherwise, allow the student to skip this intake without penalty
                                has_skipped = True
                    
                       
            
            # Add penalty for same module in consecutive intakes
            for i in range(len(chromosome[0]) - 1):  # -1 to prevent out-of-bounds error
                current_intake = {module for course in chromosome for module in course[i]}
                next_intake = {module for course in chromosome for module in course[i+1]}
                penalty_consecutive_intakes += len(current_intake.intersection(next_intake))


            
    # Add additional rewards for modules shared by certain courses
    for intake_index in range(len(chromosome[0])):
        for course_pair, shared_module_list in shared_modules.items():
            course_indices = [course_names.index(course) for course in course_pair]
            for shared_module in shared_module_list:
                if all(shared_module in chromosome[i][intake_index] for i in course_indices):
                    shared_modules_reward += 1
    
    #**not required**
    # Module inclusion reward
    # for course, modules in course_modules.items():
    #     course_index = course_names.index(course)
    #     course_chromosome = chromosome[course_index]
    #     for module in modules:
    #         if any(module in intake for intake in course_chromosome):
    #             module_inclusion_reward += 1
    #         else:
    #             module_inclusion_reward -= 5 
    
    return shared_modules_reward, first_semester_reward, penalty_students_no_module, penalty_consecutive_intakes, module_cost_penalty, total_cost


def fitness(chromosome, students_data, course_names):
    shared_modules_reward, first_semester_reward, penalty_students_no_module, penalty_consecutive_intakes, module_cost_penalty, total_cost = calc_fitness(chromosome, students_data, course_names)

    fitness_value = (shared_modules_reward * 5) + first_semester_reward - (penalty_students_no_module*10) - (penalty_consecutive_intakes*20) - (module_cost_penalty*5)

    return fitness_value,



def mutate_intake(intake, modules, prev_intake=None, next_intake=None):
    """Mutation function"""
    if len(intake) == 0: 
        return intake,
    
    # Randomly select an index in the intake
    idx = random.randint(0, len(intake) - 1)
    
    # List of possible replacement modules
    possible_modules = [m for m in modules if m not in intake]
    
    # If the previous intake and the next intake are available, ensure that modules from those intakes are not selected
    if prev_intake is not None:
        possible_modules = [m for m in possible_modules if m not in prev_intake]
    if next_intake is not None:
        possible_modules = [m for m in possible_modules if m not in next_intake]

    # If no possible replacement module, return original intake
    if not possible_modules:
        return intake,
    
    # Replace the module at the selected index with a random module from possible_modules
    if intake[idx] != "RMCE":
        intake[idx] = random.choice(possible_modules)

    return intake,


def mutate_individual(individual, modules):
    """Apply mutate_intake to all intakes in the individual."""
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            individual[i][j] = mutate_intake(individual[i][j], modules[i])[0]
    return individual,


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")
    
    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0
    
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
    
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)
    
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        # add the best back to population:
        offspring.extend(halloffame.items)
    
        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)
    
        # Replace the current population by the offspring
        population[:] = offspring
    
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    
    return population, logbook


# def get_counters(chromosome, students_data, course_names):
#     shared_modules_reward, first_semester_reward, penalty_students_no_module, penalty_consecutive_intakes, module_cost_penalty, total_cost = calc_fitness(best_individual, students_data, course_names)

#     print("Shared modules reward: ", shared_modules_reward)
#     print("First semester reward: ", first_semester_reward)
#     print("Penalty for students not taking any module: ", penalty_students_no_module)
#     print("Penalty for the same module in consecutive intakes: ", penalty_consecutive_intakes)
#     print("Module cost penalty: ", module_cost_penalty)
#     print("Total cost: ", total_cost)

def main():
    # read data
    ai = read_data(df_ai)
    ds = read_data(df_ds)
    se = read_data(df_se)
    
    students_data = {**ai, **ds, **se}
    #students_data = {**ds}

    ## Step 1: Set up the problem parameters
    population_sizes = range(50, 251, 50)
    max_generations_values = range(200, 1001, 200)
    crossover_rates = np.arange(0.5, 1.1, 0.1)
    mutation_rates = np.arange(0.5, 1.1, 0.1)
    random_seed = 42
    
    n_iterations = 1  # number of iterations to run the random search for

    results = []  # List to store the results
    
    for _ in range(n_iterations):
        population_size = random.choice(population_sizes)
        max_generations = random.choice(max_generations_values)
        p_crossover = random.choice(crossover_rates)
        m_mutation = random.choice(mutation_rates)
        
        ## Step 2: Set up the toolbox
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("individualCreator", tools.initIterate, creator.Individual, create_individual)
        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
        toolbox.register("evaluate", fitness, students_data=students_data, course_names=course_names)
        
        ## set up the genetic operators
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, modules=[modules_ai, modules_ds, modules_se])    
        
        ## set up the statistics to collect
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)
        
        ## Step 3: Set the random seed
        random.seed(random_seed)
        
        ## Step 4: Create the population
        population = toolbox.populationCreator(n=population_size)
        
        ## Hall of Fame
        hof = tools.HallOfFame(10)
        
        population, logbook = eaSimpleWithElitism(population, 
                                                  toolbox, 
                                                  cxpb=p_crossover, 
                                                  mutpb=m_mutation,
                                                  ngen=max_generations, 
                                                  stats=stats, 
                                                  halloffame=hof, 
                                                  verbose=True)
    
        # ## Step 6: Visualize the results
        # max_fitness_values = logbook.select('max')
        # avg_fitness_values = logbook.select('avg')
        # plt.plot(max_fitness_values, color='red')
        # plt.plot(avg_fitness_values, color='blue')
        # plt.xlabel('Generation')
        # plt.ylabel('Fitness')
        # plt.show()
    
        ## Step 7: Print the best schedule found
        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = fitness(best_individual, students_data, course_names)
        
        # Store the hyperparameters and the result
        results.append({
            'population_size': population_size,
            'max_generations': max_generations,
            'p_crossover': p_crossover,
            'm_mutation': m_mutation,
            'best_fitness': best_fitness,
        })

        
        
        # get_counters(best_individual, students_data, course_names)
        # courses = ['AI','DSBA','SE']
        # for x in range(len(courses)):
        #     print(courses[x])
        #     for y in range(5):
        #         print(f'Intake {y+1}: {best_individual[x][y]}')
        
    print (results)
    return results

if __name__== '__main__':
    main()

