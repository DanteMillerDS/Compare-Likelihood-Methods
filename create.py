print("CG" < "GT")


import numpy as np
import matplotlib.pyplot as plt
A=np.array([67319963484,53432057780,54306334901,47550155742,51986986870,53697169512,52421236970,52978017982,54633302343,46665415681])
B=np.array([836416698476,838920086175,830300504613,836618654498,801032517275,787951689676,784898146663,792638456036,791501039914,798649399751])

AM = np.mean(A)
print(A)
print(AM)
print(np.std(A))
BM = np.mean(B)
ASD = np.std(A)
BSD = np.std(B)
labels = ['Vectorized', 'Nonvectorized']
x_pos = np.arange(len(labels))
CTEs = [AM, BM]
error = [ASD, BSD]
print(error)
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel("Coefficient of Run Times")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Comparing Vectorized and Nonvectorized Run Times')
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()

