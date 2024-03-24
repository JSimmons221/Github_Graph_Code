import seaborn as sns
import matplotlib.pyplot as plt

plt.title("XGBoost on Repository Features")
cm = [[95, 15], [13, 62]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95)
plt.show()

plt.title("XGBoost on Graph Encodings")
cm = [[52, 58], [41, 34]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95)
plt.show()

plt.title("XGBoost on Features and Encodings")
cm = [[92, 18], [13, 62]]
sns.heatmap(cm, annot=True, vmin=13, vmax=95)
plt.show()
