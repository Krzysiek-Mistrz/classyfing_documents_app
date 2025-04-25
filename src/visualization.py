import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_loss_acc(loss_list, acc_list):
    fig, ax1 = plt.subplots()
    ax1.plot(loss_list, color="red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="red")
    ax2 = ax1.twinx()
    ax2.plot(acc_list, color="blue")
    ax2.set_ylabel("accuracy", color="blue")
    fig.tight_layout()
    plt.show()

def tsne_plot(embeddings, tokens, title="TSNE Plot"):
    tsne = TSNE(n_components=2, random_state=0)
    pts = tsne.fit_transform(embeddings)
    plt.figure()
    plt.scatter(pts[:,0], pts[:,1])
    for i, label in enumerate(tokens):
        plt.text(pts[i,0], pts[i,1], label)
    plt.title(title)
    plt.show()

def plot_embeddings_3d(embeddings, labels):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2])
    for i, lbl in enumerate(labels):
        ax.text(embeddings[i,0], embeddings[i,1], embeddings[i,2], lbl)
    plt.show()