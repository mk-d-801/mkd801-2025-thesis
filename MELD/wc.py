import string
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def disp_wordcloud(
    csv1,
    csv2,
    csv3,
    stop_word=[],
    random_seed=1,
    background="white",
    save_prefix="meld_wordcloud",
    save_dir="w_cloud"
):
    
    df = pd.concat(
        [
            pd.read_csv(csv1),
            pd.read_csv(csv2),
            pd.read_csv(csv3)
        ],
        ignore_index=True
    )

    os.makedirs(save_dir, exist_ok=True)

    emotions = df["Emotion"].dropna().unique()

    for emotion in emotions:
        df_e = df[df["Emotion"] == emotion]

        text_series = df_e["Utterance"].dropna().astype(str)
        if len(text_series) == 0:
            continue

        text = " ".join(text_series.tolist())
        text = re.sub(r"\b\w+'[a-z]+\b", lambda m: m.group(0).split("'")[0], text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = " ".join(
            word.lower()
            for word in text.split()
            if len(word) > 1
        )

        wordcloud = WordCloud(
            background_color=background,
            width=600,
            height=480,
            collocations=False,
            random_state=random_seed,
            stopwords=set(stop_word)
        ).generate(text)

        plt.figure(figsize=(10, 8))
        img = wordcloud.to_image()
        plt.imshow(img)
        plt.axis("off")

        save_path = os.path.join(
            save_dir, f"{save_prefix}_{emotion}.png"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    csv1 = "dataset/MELD.Raw/train_meld_emo.csv"
    csv2 = "dataset/MELD.Raw//dev_meld_emo.csv"
    csv3 = "dataset/MELD.Raw/test_meld_emo.csv"

    stop_words = set(STOPWORDS) | {"s", "t","Rachel","Monica","Phoebe","Ross","Chandler","Joey","guy","know","see","gonna","yeah","oh","don","okay"}

    disp_wordcloud(
        csv1,
        csv2,
        csv3,
        stop_word=stop_words,
        random_seed=42,
        background="white",
        save_prefix="MELD_wordcloud",
        save_dir="MELD/figures/w_cloud"
    )

