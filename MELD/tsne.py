import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness
import os

def load_embeddings(pt_file):
    data = torch.load(pt_file)
    embeddings = data["embeddings"]      
    labels = data["labels"]   
    return embeddings.numpy(), labels.numpy()

def plot_tsne(embeddings, labels, save_name="tsne_plot.png", perplexity=30,title="None"):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        init='random',
        random_state=42
    )

    tsne_result = tsne.fit_transform(embeddings)
    for k in [5, 10, 15]:
        tw = trustworthiness(embeddings, tsne_result, n_neighbors=k)
        print(f"k={k}, trustworthiness({title})={tw:.4f}")

    plt.figure(figsize=(10, 10))
    label_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']


    num_classes = len(label_names)
    colors = plt.colormaps.get_cmap("tab10").resampled(num_classes)


    for c in range(num_classes):
        idx = labels == c
        plt.scatter(
            tsne_result[idx, 0],
            tsne_result[idx, 1],
            s=10,
            color=colors(c),
            label=label_names[c],
            alpha=0.7
        )

    plt.legend(fontsize=14)
    plt.title(f"t-SNE of { title} Embeddings",fontsize=20)
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Saved {title} t-SNE plot to {save_name}")

def run_all_tsne(target_files, output_dir):

    for name, pt_path in target_files.items():
        save_path = os.path.join(output_dir, f"MELD_{name}_tsne.png")

        embeddings, labels = load_embeddings(pt_path)
        plot_tsne(embeddings, labels, save_path,title=name)

if __name__ == "__main__":
    pt_files = {
        "text": "./MELD/figures/embedding/pt/text_embedding.pt",
        "audio": "./MELD/figures/embedding/pt/audio_embedding.pt",
        "visual": "./MELD/figures/embedding/pt/visual_embedding.pt",
        "fusion": "./MELD/figures/embedding/pt/fusion_embedding.pt",
    }

    output_dir = "./MELD/figures/embedding/tsne/"
    run_all_tsne(pt_files, output_dir)
