import numpy as np
import pulp
import random
import sys
from tabulate import tabulate
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('WebAgg')


P_Max = 1000 #Maximale Leistung auf dem Kabel
time_span = 96 #Anzahl an Viertelstunden (96 == 24h)
Prices = [random.randrange(-10, 50, 1) for i in range(0,time_span)]
P_NVP_UNBOUND = np.linspace(start=0,stop=P_Max,num=time_span,endpoint=True) #Array mit verschiedenen Leistungswertden

#Erstellung Optimierungsmodell
model = pulp.LpProblem("XY", pulp.LpMaximize)


#Parameter Kabel
R = 50 #Ohm
U = 20000 #V


#Definition der Entscheidungsvariablen
P_BESS_NVP = \
pulp.LpVariable.dicts(
        "P_BESS_NVP",
        ('P_BESS_NVP_' + str(i) for i in range(0,time_span)),
        lowBound=0, upBound=P_Max,
        cat='Continuous')

P_BESS_NVP_V = \
pulp.LpVariable.dicts(
        "P_BESS_NVP_V",
        ('P_BESS_NVP_V_' + str(i) for i in range(0,time_span)),
        lowBound=0, upBound=P_Max,
        cat='Continuous')

P_BESS_NVP_Netto = \
pulp.LpVariable.dicts(
        "P_BESS_NVP_Netto",
        (str(i) for i in range(0,time_span)),
        lowBound=0, upBound=P_Max,
        cat='Continuous')

I_BESS_NVP = \
pulp.LpVariable.dicts(
        "I_BESS_NVP",
        ( str(i) for i in range(0,time_span)),
        lowBound=0, upBound=P_Max*1000/U,
        cat='Continuous')

I_BESS_NVP_Q = \
pulp.LpVariable.dicts(
        "I_BESS_NVP_Q",
        ( str(i) for i in range(0,time_span)),
        lowBound=0, upBound=(P_Max*1000/U)**2,
        cat='Continuous')

#Definition des Optimierungsproblems: "Maximiere die Sumem der Leistungen (abzgl. Verluste), die über die Leitung fließt"
model += pulp.lpSum(
            P_BESS_NVP_Netto[str(i)] for i in range(0,time_span))
#model += pulp.lpSum(P_BESS_NVP['P_BESS_NVP_' + str(i)]*Prices[i] for i in range(0,time_span))-pulp.lpSum(P_BESS_NVP_V['P_BESS_NVP_V_' + str(i)]*Prices[i] for i in range(0,time_span))




i_select_section = {}
min_interval = 0
max_interval = P_Max*1000/U
num_sections = 5 #Anzahl an Abschnitten
M = 1e7 #Big M
sections = {} #Lineare Abschnitte
x0_vector = np.linspace(start=min_interval,stop=max_interval,num=num_sections,endpoint=True)

#An jedem Punkt x0 der quadratischen Funktion wird eine Taylorreihe gebildet (Abbruch nach 1. Iteration)
# Bemerkung x0 bedeutet "x index 0"
#x² --> Umwandlung in Taylorreihe bei jedem x0 --> 2x0*x - x0² --> a = 2x0; b= x0²
for x0 in range(len(x0_vector)):
                sections[x0] = {}
                sections[x0]['a'] = 2*x0_vector[x0] 
                sections[x0]['b'] = -(x0_vector[x0]**2)


#Definition der Nebenbedingungen
for index in range(0,time_span):
        model += P_BESS_NVP[ 'P_BESS_NVP_' + str(index)] <= P_NVP_UNBOUND[index] 
        model += I_BESS_NVP[str(index)] == P_BESS_NVP['P_BESS_NVP_' + str(index)]*1000/U #Berechnung des Stroms 
        i_select_section[index] = \
        pulp.LpVariable.dicts(
        "i_select_section",
        ( "{}_{}".format(index,x) for x in range(len(x0_vector))),
        lowBound=0,upBound=1,cat='Binary')

        for x0 in range(len(x0_vector)):
                
                model += I_BESS_NVP_Q[str(index)] >= (sections[x0]['a']*I_BESS_NVP[str(index)]+ sections[x0]['b'])
                model += I_BESS_NVP_Q[str(index)] <= sections[x0]['a']*I_BESS_NVP[str(index)] + sections[x0]['b'] + (1-i_select_section[index]["{}_{}".format(index,x0)])*M

                model += P_BESS_NVP_V['P_BESS_NVP_V_' + str(index)]*1000 == I_BESS_NVP_Q[str(index)]*R

        #Nebenbedingung: Es darf insgesamt nur ein Abschnitt ausgewählt werden
        model +=pulp.lpSum(
                i_select_section[index]["{}_{}".format(index,lid)]
                for lid in range(len(x0_vector)))\
                == 1

        model  += P_BESS_NVP_Netto[str(index)] == P_BESS_NVP['P_BESS_NVP_' + str(index)]- P_BESS_NVP_V['P_BESS_NVP_V_' + str(index)]


model.solve() #Lösen des Problems


#Sammeln der Ergebnisse
ind_sum = {}
for index in range(0,time_span):
        ind_sum[index] = 0
        for x0 in range(len(x0_vector)): 
                ind_sum[index] += i_select_section[index]["{}_{}".format(index,x0)].varValue

table_lst = []
I_lst = []
P_V_Lst = []
Header = ["P_NVP_UNBOUND","P_BESS_NVP","Prices","P_BESS_NVP_V","I_BESS_NVP","I_BESS_NVP_Q","Summe Big M"]
for index in range(0,time_span):
        table_lst.append([P_NVP_UNBOUND[index],P_BESS_NVP['P_BESS_NVP_' +str(index)].varValue,Prices[index],P_BESS_NVP_V['P_BESS_NVP_V_' + str(index)].varValue,I_BESS_NVP[str(index)].varValue,I_BESS_NVP_Q[str(index)].varValue,str(ind_sum[index])])
        I_lst.append(I_BESS_NVP[str(index)].varValue)
        P_V_Lst.append(P_BESS_NVP_V['P_BESS_NVP_V_' + str(index)].varValue)
print(tabulate(table_lst, headers=Header, tablefmt='fancy_grid'))



curr_dir = os.path.dirname(os.path.realpath(__file__)) #Aktueller Ordnerpfad
result_path = os.path.join(curr_dir,"Approx_Test.xlsx")

if os.path.exists(result_path):
        os.remove(result_path)

x0 = np.array(I_lst)
y = np.array(P_V_Lst)
y_real = []
#Berechnung von "echtem Ergebnis"
for nr, P in enumerate(P_NVP_UNBOUND):
        I = P*1000/U
        P_V = I**2 * R/1000
        y_real.append(P_V)


#Plotten von Approxomierter und Quadratischer Lösung
plt.plot(x0,y,x0,y_real)
#plt.plot(x,y)
plt.show()

df = pd.DataFrame (table_lst, columns = Header)
df.to_excel(result_path,sheet_name = 'Tabelle1',index = False,engine = 'xlsxwriter')