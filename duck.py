import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

cm = [[95, 15], [13, 62]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95)
plt.savefig('RepoFeats.svg', format='svg')
plt.show()

cm = [[52, 58], [41, 34]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95)
plt.savefig('Encode.svg', format='svg')
plt.show()

cm = [[92, 18], [13, 62]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95)
plt.savefig('FeatsEncode.svg', format='svg')
plt.show()
