import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
"""
Created on Mon Jan 01 10:01:15 2024
"""
# Définition de la méthode de Lagrange pour l'interpolation polynomiale
def lagrange_interpolation(x, y, xi):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term = term * (xi - x[j]) / (x[i] - x[j])
        result += term
    return result

# Définition de la méthode de Newton pour l'interpolation polynomiale
def newton_interpolation(x, y, xi):
    n = len(x)
    coefficients = np.zeros(n)

    # Calcul des différences divisées
    for i in range(n):
        coefficients[i] = y[i]
    
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            coefficients[j] = (coefficients[j] - coefficients[j - 1]) / (x[j] - x[j - i])


    # Évaluation du polynôme de Newton à xi
    result = coefficients[-1]
    for i in range(n - 2, -1, -1):
        result = result * (xi - x[i]) + coefficients[i]

    return result
# Fonction pour créer des points de données manuellement
def manually_enter_data_points(num_points):
    x = []
    y = []
    for i in range(num_points):
        x_i = float(input(f"Entrez la valeur de x_{i+1} : "))
        y_i = float(input(f"Entrez la valeur de y_{i+1} : "))
        x.append(x_i)
        y.append(y_i)
    return np.array(x), np.array(y)

# Fonction pour créer des points de données de manière aléatoire
def generate_data_points(num_points):
    x = np.sort(np.random.uniform(-10, 10, num_points))
    y = np.sin(x) + np.random.normal(0, 0.1, num_points)
    return x, y

# Fonction principale pour effectuer l'interpolation et afficher les résultats
def main():
    try:
        num_points = int(input("Entrez le nombre de points de données : "))
        
        x, y = manually_enter_data_points(num_points)

        xi_range = np.linspace(min(x), max(x), 100) 

        # Utilisation du module multiprocessing pour effectuer les interpolations en parallèle
        with Pool(processes=2) as pool:
            lagrange_results = pool.starmap(lagrange_interpolation, [(x, y, xi) for xi in xi_range])
            newton_results = pool.starmap(newton_interpolation, [(x, y, xi) for xi in xi_range])

        # Tracé des points de données et des interpolations
        plt.scatter(x, y, label='Points de données')
        plt.plot(xi_range, lagrange_results, color='g', label='Interpolation de Lagrange')
        plt.plot(xi_range, newton_results, color='b', label='Interpolation de Newton')

        plt.legend()
        plt.show()

        # Affichage du tableau des valeurs x et y
        print("Tableau des valeurs de x et y :")
        print(np.column_stack((x, y)))

    except ValueError as ve:
        print(f"Erreur : {ve}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")

if __name__ == "__main__":
    main()
