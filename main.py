import streamlit as st 
import numpy as np
import random
from pymoo.core.problem import ElementwiseProblem

import numpy as np
def generate_random_fog_values(n):
    cpu = []
    cost = []
    for i in range(n):
        execution = random.randint(40000, 150000)
        price = execution * 0.0001
        cpu.append(execution)
        cost.append(price)
    return np.array(cpu), np.array(cost)

def generate_random_task_values(n):
    values = []
    for i in range(n):
        execution = random.randint(2000000, 30000000)
        values.append(execution)
    return np.array(values)


class TaskServerProblem(ElementwiseProblem):
    def __init__(self, n_tasks, n_servers, n_obj, task_instructions, server_cpu, server_costs=None):
        super().__init__(n_var=n_tasks, n_obj=n_obj, xl=0, xu=n_servers-1)
        self.n_tasks = n_tasks
        self.n_servers = n_servers
        self.task_times = np.random.rand(self.n_tasks)
        self.task_instructions =  task_instructions
        #self.server_cpu, self.server_costs = generate_random_fog_values(n_servers) #cpu et coût des serveurs
        self.server_cpu = server_cpu
        self.server_costs = server_costs
        

    def _evaluate(self, x, out, *args, **kwargs):
        # fonction objectif 1 : minimiser le temps total d'exécution des tâches
        x = np.round(x).astype(int) #liste solution
        
        server_times = np.zeros(self.n_servers) # temps d'exécution total pour chaque serveur
        for i in range(self.n_tasks):
          #server_times[x[i]] += self.task_times[i]
          if (server_times[x[i]] <  (self.task_instructions[i]/self.server_cpu[x[i]])).any():
            server_times[x[i]] = self.task_instructions[i]/self.server_cpu[x[i]]

        f1 = np.max(server_times)

        # fonction objectif 2 : minimiser le coût total d'exécution
        f2 = 0
        for i in range(self.n_tasks):
          f2 += self.task_instructions[i]*self.server_costs[x[i]] #coût total d'exécution de toutes les tâches

        # fonction objectif 3 : maximiser le nombre de ressources utilisés
        count_serveur = set(x)
        f3 = len(count_serveur)
        
        #out["F"] = [f1,f2,f3]
        out["F"] = [f1,f2]    
        if(self.n_obj == 3) :
            out["F"].append(f3)

        # contrainte : chaque tâche doit être attribuée à un seul serveur
        #out["G"] = np.max(np.bincount(x, minlength=self.n_servers))-1

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.visualization.scatter import Scatter
import itertools
st.title("Démonstration de notre projet de meta-heuristiques: Task Scheduling")

#Utilisation du formalisme Latex pour présenter le projet sur notre page interactive
st.header('Enoncé du problème')
st.write('On souhaite allouer un nombre de tâches définies sur différentes unités de calcul. L’objectif est de répartir la charge sur ces différents serveurs de façon à minimiser le temps d éxécution, le coût d exécution et l’utilisation des ressources.')
st.header('Modélisation mathématique')
st.write('\U0001F3AF Objectif n°1: Minimiser le temps d execution. ')
st.latex('Min (Max(Ti)) \quad  où \ Ti \ représente \  une \  ième \ tache. ')
st.write('\U0001F3AF Objectif n°2: Minimiser le cout total')
st.latex('Min(\sum_{i=1}^{n} T_i \cdot C_{vm_i})  \quad  où \ Ti \ représente \  une \  ième \ tache \ et \ Cvmi \ le \ cout \ de \ la \ ieme \ vm')
st.write('\U0001F3AF Objectif n°3: Minimiser le nombre de ressources utilisés (le nombre de serveurs) ')
st.latex('Min (Rs) \quad  où \ Rs \ représente \  les  \  ressources. ')
st.header('Paramètres du probleme')

#Input pour determiner le nombre de taches voulus, nombre de serveurs voulus et choix du type de valeurs
n_tasks = st.number_input('Saisir le nombre de tâches', value=50, min_value=1, max_value=50)
n_servers = st.number_input('Saisir le nombre de serveurs', value=5, min_value=2, max_value=5)
donnees =  st.radio('A: Valeurs par défaut | B: Valeurs au hasard', ['A', 'B'])

if donnees == 'A':
    task_instructions=np.array([3324661,6725193,13723233,5779370,14563959,10202920,18713183,16684225,15556916,9946119,15537110,22322498,14681901,22989230,23444017,13560735,8833867,6920224,13022243,12300466])
    server_cpu=np.array([95546,100505,44292,117400,110803])[0:n_servers]
    server_costs=np.array([3.46478974,5.69474934,1.9086627,6.94341464,5.03096405])[0:n_servers]
    pass
else:
    task_instructions=generate_random_task_values(n_tasks)
    server_cpu, server_costs = generate_random_fog_values(n_servers) #cpu et coût des serveurs
    print(server_cpu)
    print(server_costs)
    pass
st.header('Paramètres de NSGA2')
#Choix des paramètres de NSGA2
pop_size = st.number_input('Saisir la population souhaité', value=100)
objectif =  st.radio('A: Minimiser temps déxecution et le cout déxecution | B: Minimiser temps déxecution, le cout déxecution et le nombre de serveurs à solliciter', ['A', 'B'])
sampling = st.radio('Sampling', [BinaryRandomSampling, LatinHypercubeSampling])
crossover = st.radio('Crossover', [TwoPointCrossover, UniformCrossover])
mutation = st.radio('Mutation', [BitflipMutation, GaussianMutation])

if st.button('Lancer NSGA2'):
    if objectif == 'A':
        n_obj = 2
        problem = TaskServerProblem(n_tasks=n_tasks, n_servers=n_servers,task_instructions=task_instructions, server_cpu=server_cpu, server_costs=server_costs, n_obj=n_obj)
        st.write("Vous avez choisi d'optimiser à la fois le temps d'exécution et le coût d'exécution, mais pas le nombre de serveurs à solliciter.")
    else:
        n_obj = 3
        problem = TaskServerProblem(n_tasks=n_tasks, n_servers=n_servers, task_instructions=task_instructions, server_cpu=server_cpu, server_costs=server_costs, n_obj=n_obj)
        st.write("Vous avez choisi de maintenir les 3 objectifs.")
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling(),
        crossover=crossover(),
        mutation=mutation(),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 100)
    print(termination)
    res = minimize(problem, algorithm, termination,seed=1,verbose=False)

    print(res.X) # attribution des tâches à chaque serveur
    print(res.F) # [temps total d'exécution des tâches, coût

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import plotly.graph_objs as go

    # res_data = res.F.T
    # x = res_data[0]
    # y = res_data[1]
    # z = res_data[2]

    # # Afficher le graphique Matplotlib
    # # st.write("Graphique Matplotlib")
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(x, y, z, c='r', marker='o')
    # # ax.set_xlabel('X')
    # # ax.set_ylabel('Y')
    # # ax.set_zlabel('Z')
    # # st.pyplot(fig)


    if (n_obj == 2):
        #Afficher le graphique 2d
        res_data = res.F.T
        fig = go.Figure(data=go.Scatter(x=res_data[0],y=res_data[1], mode='markers'))
        st.plotly_chart(fig)
    else:
        #Afficher le graphique 3d

        st.header('Représentation graphique des solutions pareto')
        res_data = res.F.T
        x = res_data[0]
        y = res_data[1]
        z = res_data[2]
        fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, mode='markers'))
        st.plotly_chart(fig)
        fig2 = go.Figure(data=go.Scatter(x=res_data[0],y=res_data[2], mode='markers'))
        fig2.update_layout(
        xaxis_title="temps en secondes",
        yaxis_title="nombres de serveurs",
        title="Nombre de sollutions sollicités par le temps d'exeuction"
        )
        st.plotly_chart(fig2)

        
