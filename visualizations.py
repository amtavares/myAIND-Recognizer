import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('results1.txt')

df3 = df.pivot(index='feature', columns='modelselector', values='wer')



print(df3)
print('\n\n')


ax = sns.heatmap(df3 ,annot=True, cmap="YlGnBu")

# turn the axis label
# for item in ax.get_yticklabels():
#     item.set_rotation(0)

# for item in ax.get_xticklabels():
#     item.set_rotation(90)
plt.tight_layout()

plt.savefig('results1.png')

plt.show()