import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

gamma = 0.5
alpha = 0.3
n = 4

r_list = [-2.0, 4.0, 1.0] #rewards - constant

epochs = 25
q_original = [1, 2, 8] #starting point

true_q = np.zeros(n-1)

cur = 0

for j in range(len(true_q)-1, -1):
    true_q[j] = r_list[j] + gamma*cur
    cur = true_q[j]

q_table = np.zeros([epochs, n])

for j in range(n-1):
    q_table[0, j] = q_original[j]

for x0 in range(1, epochs):
    for x1 in range(n-1):
        learned = r_list[x1] + gamma * q_table[x0-1, x1+1] - q_table[x0-1, x1]
        q_table[x0, x1] = q_table[x0-1, x1]+ alpha * learned   


fig, ax = plt.subplots(1, 1, figsize = (5,3), dpi = 200)
colors = ['#2CA02c', '#FF7F0E', '#D62778']
markers = ['o', 'd', '^']

for j in range(n-1):
    ax.plot(np.arange(epochs), q_table[:, j], marker = markers[j], markersize = 5,
            alpha = 0.7, color = colors[j], linestyle = '-',  label = f'$Q$'+f's{j+1}')
    ax.axhline(y = true_q[j], color = colors[j], linestyle = '--')

ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_ylabel('Q-values')
ax.set_xlabel('episode')
ax.set_title(r'$\gamma=$'+f'{gamma}'+r'$\alpha=$'+f'{alpha}')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

