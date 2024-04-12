import seaborn as sns
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({'font.size': 28})

cm = [[95, 15], [13, 62]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95, cmap="YlGnBu")
plt.savefig('RepoFeats.svg', format='svg')
plt.show()

cm = [[52, 58], [41, 34]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95, cmap="YlGnBu")
plt.savefig('Encode.svg', format='svg')
plt.show()

cm = [[92, 18], [13, 62]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95, cmap="YlGnBu")
plt.savefig('FeatsEncode.svg', format='svg')
plt.show()

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
save_dir = './result'
print(device)
