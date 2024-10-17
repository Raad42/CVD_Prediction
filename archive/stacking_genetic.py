import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv('UCI Cardiovascular.csv')
data.replace(['?', '-9'], np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())

data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)
data_sampled = data.sample(frac=1, random_state=42)
X = data_sampled.drop(columns=['num'])
y = data_sampled['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Define Base Models (Level-0) ###
knn = KNeighborsClassifier()
svm = SVC(probability=True, random_state=42)
dt = DecisionTreeClassifier(random_state=42)

### Define Meta-Model (Level-1) ###
meta_model = LogisticRegression()

# Create Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[
        ('knn', knn),
        ('svm', svm),
        ('dt', dt)
    ],
    final_estimator=meta_model,
    cv=5
)

### Genetic Algorithm Setup ###
population_size = 10
generations = 20
mutation_rate = 0.1

# Define parameter ranges
param_ranges = {
    'knn__n_neighbors': [3, 5],
    'svm__C': [0.1, 1],
    'svm__kernel': ['linear', 'rbf'],
    'dt__max_depth': [None, 5],
    'final_estimator__C': [0.1, 1]
}

# Function to create a random individual (parameter set)
def random_individual():
    individual = {
        'knn__n_neighbors': random.choice(param_ranges['knn__n_neighbors']),
        'svm__C': random.choice(param_ranges['svm__C']),
        'svm__kernel': random.choice(param_ranges['svm__kernel']),
        'dt__max_depth': random.choice(param_ranges['dt__max_depth']),
        'final_estimator__C': random.choice(param_ranges['final_estimator__C'])
    }
    return individual

# Function to evaluate fitness (AUC score)
def evaluate_fitness(individual):
    knn = KNeighborsClassifier(n_neighbors=individual['knn__n_neighbors'])
    svm = SVC(probability=True, C=individual['svm__C'], kernel=individual['svm__kernel'], random_state=42)
    dt = DecisionTreeClassifier(max_depth=individual['dt__max_depth'], random_state=42)
    meta_model = LogisticRegression(C=individual['final_estimator__C'])
    
    stacking_model = StackingClassifier(
        estimators=[('knn', knn), ('svm', svm), ('dt', dt)],
        final_estimator=meta_model,
        cv=5
    )
    
    stacking_model.fit(X_train, y_train)
    y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc

# Create initial population
population = [random_individual() for _ in range(population_size)]

# Genetic Algorithm Loop
best_individual = None
best_fitness = 0

for generation in range(generations):
    print(f"Generation {generation+1}/{generations}")
    
    # Evaluate the fitness of each individual
    fitness_scores = [evaluate_fitness(ind) for ind in population]
    
    # Select the best individual
    for i, fitness in enumerate(fitness_scores):
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = population[i]
    
    # Tournament selection
    selected_population = []
    for _ in range(population_size // 2):
        ind1, ind2 = random.sample(population, 2)
        if evaluate_fitness(ind1) > evaluate_fitness(ind2):
            selected_population.append(ind1)
        else:
            selected_population.append(ind2)
    
    # Crossover
    new_population = []
    for i in range(0, len(selected_population), 2):
        parent1 = selected_population[i]
        parent2 = selected_population[(i+1) % len(selected_population)]
        child1 = {k: parent1[k] if random.random() > 0.5 else parent2[k] for k in parent1}
        child2 = {k: parent2[k] if random.random() > 0.5 else parent1[k] for k in parent1}
        new_population.extend([child1, child2])
    
    # Mutation with fuzzy logic (adjust based on fuzzy rules)
    for individual in new_population:
        if random.random() < mutation_rate:
            # Apply fuzzy mutation rule (e.g., if fitness is low, apply larger mutation)
            individual['knn__n_neighbors'] = random.choice(param_ranges['knn__n_neighbors'])
            individual['svm__C'] = random.choice(param_ranges['svm__C'])
    
    # Update population
    population = new_population

# Final evaluation of the best individual
print(f"Best Individual: {best_individual}")
print(f"Best AUC Score: {best_fitness:.2f}")

# Plot ROC Curve for the best individual
best_model = StackingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=best_individual['knn__n_neighbors'])),
        ('svm', SVC(probability=True, C=best_individual['svm__C'], kernel=best_individual['svm__kernel'], random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=best_individual['dt__max_depth'], random_state=42))
    ],
    final_estimator=LogisticRegression(C=best_individual['final_estimator__C']),
    cv=5
)

best_model.fit(X_train, y_train)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"Best Model (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Best Genetic Fuzzy Stacking Model')
plt.legend(loc='lower right')
plt.show()
