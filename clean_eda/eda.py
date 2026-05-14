#!/usr/bin/env python3

"""
Exploratory Data Analysis module for mental-health text classification.

Depends on:
    A Cleaning instance for the NLP model, stopword set, and regex maps.
    :class:PostAnalysis (Pydantic model) for typed feature rows.
"""

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import emoji
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from wordcloud import WordCloud
from venn import venn

from .cleaning import PostAnalysis

if TYPE_CHECKING:
    from .cleaning import Cleaning

warnings.filterwarnings("ignore")


class EDA:
    """
    Concrete class responsible for feature extraction, linguistic analysis,
    and visualisation of mental-health social-media data.

    :param cleaner: Instantiated :class:`Cleaning`
        object, supplying the spaCy model, stopword set, and regex
        maps.
    :type cleaner: Cleaning
    :param output_dir: Directory where plots and summary files are
        written.
    :type output_dir: str
    """

    #: Label of hex colors used consistently across all plots.
    _PALETTE: Dict[str, str] = {
        "Anxiety":    "#FF6B6B",
        "Depression": "#463EBD",
        "Suicidal":   "#55D7BF",
        "Normal":     "#D8BE3D",
    }

    def __init__(
        self,
        cleaner: "Cleaning",
        output_dir: str = "plots",
    ) -> None:
        """
        Initialise the EDA instance.

        :param cleaner: Shared :class:Cleaning
            instance (provides nlp, stopwords_set,
            emojis_map, emoticons_map).
        :type cleaner: Cleaning
        :param output_dir: Folder for saved plots and summary text.
        :type output_dir: str
        """
        self.cleaner = cleaner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.dpi"] = 120

    # FEATURE EXTRACTION

    def count_punctuation(self, text: str) -> Dict[str, int]:
        """
        Count ?, !, and ... in text.

        :param text: Cleaned post string.
        :type text: str
        :returns: Mapping of punctuation type to count.
        :rtype: Dict[str, int]
        """
        return {
            "question_count":    text.count("?"),
            "exclamation_count": text.count("!"),
            "ellipsis_count":    text.count("..."),
        }

    def extract_hashtags(self, text: str) -> Tuple[List[str], str]:
        """
        Extract hashtags from text and return them separately.

        :param text: Input string.
        :type text: str
        :returns: A tuple of ``(hashtag_list, text_without_hashtags).
        :rtype: Tuple[List[str], str]
        """
        hashtags: List[str] = re.findall(r"#\w+", text)
        text = re.sub(r"#\w+", " ", text)
        return hashtags, text

    def detect_emojis_combined(
        self, text: str
    ) -> Tuple[List[str], int]:
        """
        Detect Unicode emojis using the :mod:emoji library with a
        regex fallback to ensure full coverage.

        :param text: Raw (original) post string.
        :type text: str
        :returns: Tuple of (demojized_name_list, count).
        :rtype: Tuple[List[str], int]
        """
        found = [item["emoji"] for item in emoji.emoji_list(text)]
        # as many emojis do not render in matplotlib
        standardized = [emoji.demojize(e) for e in found]
        # regex fallback to ensure full coverage
        for match in re.findall(
            self.cleaner.emojis_map, text, re.UNICODE
        ):
            name = emoji.demojize(match)
            if name not in standardized:  # make sure of no duplicates
                standardized.append(name)
        return standardized, len(standardized)

    def detect_emoticons(
        self, text: str
    ) -> Tuple[List[str], int]:
        """
        Detect ASCII emoticons in text using the project regex list.

        :param text: Raw (original) post string.
        :type text: str
        :returns: Tuple of ``(matched_emoticon_list, count)``.
        :rtype: Tuple[List[str], int]
        """
        found: List[str] = []
        for pattern in self.cleaner.emoticons_map:
            found.extend(
                re.findall(pattern, text, re.IGNORECASE)
            )
        return found, len(found)

    def extract_ngrams(
        self, tokens: List[str], n: int
    ) -> List[str]:
        """
        Build n-grams from a token list, keeping only all-alphabetic
        ones.

        :param tokens: Lemmatised, stop-word-free token list.
        :type tokens: List[str]
        :param n: n-gram size (1 = unigram, 2 = bigram, etc.).
        :type n: int
        :returns: List of n-gram strings.
        :rtype: List[str]
        """
        if len(tokens) < n:
            return []
        # Sliding window that generates n-grams
        # and filter out those containing non-alphabetic tokens
        # ["i", "feel", "sad"]
        # bigrams → ["i feel", "feel sad"]
        raw = [
            " ".join(tokens[i: i + n])
            for i in range(len(tokens) - n + 1)
        ]
        # filter out n-grams that contain any non-alphabetic token (e.g. can 2)
        return [
            ng for ng in raw
            if all(w.isalpha() for w in ng.split())
        ]

    def compute_post_features_from_doc(
        self,
        original_text: str,
        cleaned_text: str,
        doc: object,
    ) -> PostAnalysis:
        """
        Compute all post-level features from a pre-built spaCy doc.

        :param original_text: Raw post (used for emoji/emoticon
            detection).
        :type original_text: str
        :param cleaned_text: Normalised post (used for punctuation
            counts and char count).
        :type cleaned_text: str
        :param doc: spaCy Doc produced by nlp.pipe().
        :type doc: spacy.tokens.Doc
        :returns: Fully populated :class:PostAnalysis instance.
        :rtype: PostAnalysis
        """
        # Word count / Text length from the pre-computed doc
        # (only counting alphabetic tokens)
        word_count = sum(1 for token in doc if token.is_alpha)
        char_count = len(cleaned_text)

        punct_counts = self.count_punctuation(cleaned_text)
        total_punct = sum(punct_counts.values())
        # normalize long vs short posts
        punct_density = (
            total_punct / word_count if word_count > 0 else 0.0
        )

        emojis, emoji_count = self.detect_emojis_combined(
            original_text
        )
        emoticons, emoticon_count = self.detect_emoticons(
            original_text
        )

        hashtags, _ = self.extract_hashtags(original_text)

        return PostAnalysis(
            text_length=word_count,
            char_count=char_count,
            punct_density=punct_density,
            question_count=punct_counts.get("question_count", 0),
            exclamation_count=punct_counts.get(
                "exclamation_count", 0
            ),
            ellipsis_count=punct_counts.get("ellipsis_count", 0),
            emoji_count=emoji_count,
            emojis=emojis,
            emoticon_count=emoticon_count,
            emoticons=emoticons,
            hashtags=hashtags,
        )

    # Featire extraction entry point method
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all features to df via an nlp.pipe() pass.

        :param df: dataframe containing the posts
        :type df: pd.DataFrame
        :return: new dataframe adding new features / columns to the old one.
        :rtype: pd.DataFrame
        """
        print("\nEXTRACTING TEXT FEATURES...")
        print(f"Processing {len(df):,} samples with spaCy...")

        cleaner = self.cleaner

        # Reuse cached docs from clean_dataframe
        if hasattr(cleaner, "_cached_docs") and len(cleaner._cached_docs) == len(df):
            docs = cleaner._cached_docs
            print("  Using cached spaCy docs from cleaning step.")
        else:
            # Fallback if cache is missing
            docs = list(tqdm(
                cleaner.nlp.pipe(df["cleaned_text"].tolist(), batch_size=500),
                total=len(df),
                desc="  spaCy NLP pass (fallback)",
                dynamic_ncols=True,
            ))

        feature_data: List[Dict] = []
        ttr_values: List[float] = []

        original_texts = df["text"].values
        cleaned_texts = df["raw_cleaned_text"].values

        for i, doc in enumerate(tqdm(
            docs, total=len(df), desc="  Extracting features"
        )):
            analysis = self.compute_post_features_from_doc(
                original_texts[i], cleaned_texts[i], doc
            )
            feature_data.append(analysis.model_dump())
            ttr_values.append(self._compute_ttr(df["tokens"].iloc[i]))

        features_df = pd.DataFrame(feature_data).drop(columns=["hashtags"])
        df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        df["ttr"] = ttr_values

        # Clear cache after use
        if hasattr(cleaner, "_cached_docs"):
            del cleaner._cached_docs

        print(f"Features extracted. Shape: {df.shape}")
        return df

    # ANALYSIS USED FOR PLOTTING

    def _compute_ttr(self, tokens: List[str]) -> float:
        """
        Compute the type-token ratio for a list of tokens.

        :param tokens: Lemmatised token list for one post.
        :type tokens: List[str]
        :returns: TTR in [0, 1]; 0.0 for empty input.
        :rtype: float
        """
        if not tokens:
            return 0.0
        # sets allow for only unique tokens
        # so we can count how many unique tokens there are
        return len(set(tokens)) / len(tokens)

    def compute_normalized_punctuation(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Average punctuation counts per sentence, grouped by label.

        :param df: Feature DataFrame with text, status,
            question_count,`exclamation_count, and
            ellipsis_count columns.
        :type df: pandas.DataFrame
        :returns: DataFrame indexed by label with one column per
            punctuation type, values normalised by mean sentence count.
        :rtype: pandas.DataFrame
        """
        punct_cols = [
            "question_count", "exclamation_count", "ellipsis_count"
        ]
        # split text into sentences
        # avoid division by zero, treat empty text as 1 sentence
        sentence_counts = df["text"].apply(
            lambda t: max(len(sent_tokenize(t)), 1)
        )
        summary_data = []
        # for each label we compute the average punctuation count per sentence
        # by dividing the average punctuation count
        # by the average sentence count for that label
        for label in df["status"].unique():
            # create a dataframe for the current label
            label_data = df[df["status"] == label]
            avg_sent = sentence_counts[label_data.index].mean()
            row: Dict[str, object] = {"Label": label}
            for col in punct_cols:
                avg = label_data[col].mean()
                # exclamation_count -> Exclamation
                row[col.replace("_count", "").title()] = (
                    avg / avg_sent if avg_sent > 0 else 0.0
                )
            # save result into the list
            summary_data.append(row)
        return (
            pd.DataFrame(summary_data).set_index("Label")
        )

    def compute_label_top_ngrams(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        n: int,
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Return the top_N most frequent n-grams for every label.

        :param df: Feature DataFrame.
        :type df: pandas.DataFrame
        :param text_column: Column containing text to n-gram if tokens
            are unavailable.
        :type text_column: str
        :param label_column: Column containing class labels.
        :type label_column: str
        :param n: n-gram size.
        :type n: int
        :param top_k: How many top n-grams to return per label.
        :type top_k: int
        :returns: {label: [(ngram, count), …]}
        :rtype: Dict[str, List[Tuple[str, int]]]
        """
        # Example output: {'Anxiety': [('feel sad', 15), ('not happy', 10)],
        # 'Depression': [('feel hopeless', 20), ...], ...}
        label_ngrams: Dict[str, List[Tuple[str, int]]] = {}

        for label in df[label_column].unique():
            mask = df[label_column] == label
            # Counter is a dict subclass for counting hashable objects.
            # It counts the frequency of each n-gram.
            counter: Counter = Counter()

            # Use cached tokens if available
            if "tokens" in df.columns:
                # in this case, we already have the list of words
                for tokens in df[mask]["tokens"]:
                    # some rows might have missing or malformed tokens,
                    # so we check if it's a list before processing
                    if isinstance(tokens, list):
                        counter.update(
                            self.extract_ngrams(tokens, n)
                        )
            else:
                # We don't have the list
                for text in df[mask][text_column]:
                    tokens = self.cleaner.nlp(text)
                    lemmas = [
                        t.lemma_.lower()
                        for t in tokens
                        if t.is_alpha
                        and t.lemma_.lower()
                        not in self.cleaner.stopwords_set
                    ]
                    counter.update(self.extract_ngrams(lemmas, n))

            label_ngrams[label] = counter.most_common(top_k)

        return label_ngrams

    def compute_shared_vocabulary(
        self, df: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """
        Build a unique-word vocabulary set for each label.

        :param df: Feature DataFrame
        :type df: pandas.DataFrame
        :returns: {label: set_of_unique_words}
        :rtype: Dict[str, Set[str]]
        """
        label_vocab: Dict[str, Set[str]] = {}

        for label in df["status"].unique():
            mask = df["status"] == label
            vocab: Set[str] = set()

            # Use cached tokens if available
            if "tokens" in df.columns:
                for tokens in df[mask]["tokens"]:
                    if isinstance(tokens, list):
                        vocab.update(
                            t for t in tokens if t and t.isalpha()
                        )
            else:
                col = (
                    "cleaned_text"
                    if "cleaned_text" in df.columns
                    else "text"
                )
                for text in df[mask][col]:
                    if not isinstance(text, str):
                        continue
                    doc = self.cleaner.nlp(text)
                    lemmas = [
                        t.lemma_.lower()
                        for t in doc
                        if t.is_alpha
                        and t.lemma_.lower()
                        not in self.cleaner.stopwords_set
                    ]
                    vocab.update(
                        t for t in lemmas if t and t.isalpha()
                    )

            label_vocab[label] = vocab

        return label_vocab

    # VISUALISATION

    def _colour(self, label: str) -> str:
        """Return the hex colour for label, defaulting to grey."""
        return self._PALETTE.get(label, "#999999")

    def plot_label_distribution(self, df: pd.DataFrame) -> None:
        """
        Pie chart of label counts.

        :param df: Feature DataFrame with a status column.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        counts = df["status"].value_counts()
        colors = [self._colour(lbl) for lbl in counts.index]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 12, "weight": "bold"},
        )
        ax.set_title(
            "Mental Health Label Distribution",
            fontsize=14, weight="bold", pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "label_distribution_pie.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def plot_text_length_distribution(
        self, df: pd.DataFrame
    ) -> None:
        """
        2 × 2 grid of word-count histograms with median lines, one per
        label.

        :param df: Feature DataFrame with ``text_length`` and
            ``status`` columns.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for idx, label in enumerate(df["status"].unique()):
            lengths = df[df["status"] == label]["text_length"]
            axes[idx].hist(
                lengths, bins=30,
                color=self._colour(label),
                edgecolor="black", alpha=0.7,
            )
            median = lengths.median()
            axes[idx].axvline(
                median, linestyle="--", color="blue",
                linewidth=2, label=f"Median: {median:.1f}",
            )
            axes[idx].set_title(
                f"Text Length Distribution – {label}",
                fontsize=12, weight="bold",
            )
            axes[idx].set_xlabel("Text Length", fontsize=10)
            axes[idx].set_ylabel("Frequency",   fontsize=10)
            axes[idx].grid(axis="y", alpha=0.3)
            axes[idx].legend()
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "text_length_histograms.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def plot_individual_boxplots(
        self,
        df: pd.DataFrame,
        column: str,
        ylabel: str,
        filename_prefix: str,
    ) -> None:
        """
        One boxplot per label for the word_count/text_length.

        :param df: Feature DataFrame.
        :type df: pandas.DataFrame
        :param column: Column to plot on the y-axis.
        :type column: str
        :param ylabel: Y-axis label text.
        :type ylabel: str
        :param filename_prefix: Prefix for output filenames (label name
            is appended automatically).
        :type filename_prefix: str
        :returns: None
        :rtype: None
        """
        for label in df["status"].unique():
            data = df[df["status"] == label][column]
            fig, ax = plt.subplots(figsize=(8, 6))
            bp = ax.boxplot(
                [data], labels=[label],
                patch_artist=True, widths=0.5, vert=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(self._colour(label))
                patch.set_alpha(0.8)
            ax.set_ylabel(ylabel, fontsize=12, weight="bold")
            ax.set_title(
                f"{ylabel} Distribution – {label}",
                fontsize=14, weight="bold", pad=20,
            )
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"{filename_prefix}_{label}.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close()

    def plot_punctuation_normalized_table(
        self, summary_df: pd.DataFrame
    ) -> None:
        """
        Render the normalised punctuation summary as a styled table.

        :param summary_df: Output of
            :meth:compute_normalized_punctuation.
        :type summary_df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=summary_df.round(4).values,
            rowLabels=summary_df.index,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
            colWidths=[0.25] * len(summary_df.columns),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(
                weight="bold", color="white"
            )
        for i, label in enumerate(summary_df.index):
            table[(i + 1, -1)].set_facecolor(self._colour(label))
            table[(i + 1, -1)].set_text_props(
                weight="bold", color="white"
            )
        plt.title(
            "Punctuation Usage Normalized by Sentence Length",
            fontsize=14, weight="bold", pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "punctuation_normalized_table.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def plot_wordclouds(self, df: pd.DataFrame) -> None:
        """
        One word cloud per label, using the cached tokens column
        when available.

        :param df: Feature DataFrame.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        for label in df["status"].unique():
            mask = df["status"] == label

            # Use cached tokens instead of re-running preprocess()
            if "tokens" in df.columns:
                combined = " ".join(
                    " ".join(toks)
                    for toks in df[mask]["tokens"]
                    if isinstance(toks, list)
                )
            else:
                # Fallback if tokens not cached
                col = (
                    "cleaned_text"
                    if "cleaned_text" in df.columns
                    else "text"
                )
                combined = " ".join(
                    " ".join(t for t in toks
                             if t not in self.cleaner.wordcloud_stopwords_set)
                    for toks in df[mask]["tokens"]
                    if isinstance(toks, list)
                )

            wc = WordCloud(
                width=1200, height=600,
                background_color="white",
                colormap="tab20c",
                stopwords=self.cleaner.wordcloud_stopwords_set,
                min_font_size=12,
                max_words=80,
                collocations=False,
                # ensures that only single words are included
                # not bigrams or trigrams
            ).generate(combined)

            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(
                f"Word Cloud – {label}",
                fontsize=16, weight="bold", pad=20,
            )
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(
                self.output_dir
                / f"wordcloud_standalone_{label}.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close()

    def plot_status_word_venn(
        self, label_vocab: Dict[str, Set[str]]
    ) -> None:
        """
        Four-set Venn diagram of word overlap across labels.

        Uses the pre-computed vocabulary from :meth:compute_shared_vocabulary
        to ensure consistency with vocabulary statistics table.

        :param label_vocab: Output of :meth:compute_shared_vocabulary.
        :type label_vocab: Dict[str, Set[str]]
        :returns: None
        :rtype: None
        """
        status_words = {
            label: vocab
            for label, vocab in label_vocab.items()
            if label in ["Depression", "Anxiety", "Suicidal", "Normal"]
        }

        fig = plt.figure(figsize=(12, 10))
        venn(status_words)
        plt.title(
            "Status Word Overlap Venn Diagram",
            fontsize=16, weight="bold", pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "status_word_overlap_venn.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    def plot_label_ngrams(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        top_k: int = 10,
    ) -> None:
        """
        Bar charts for top unigrams, bigrams, and trigrams per label
        (12 plots total: 4 labels × 3 n-gram sizes).

        :param df: Feature DataFrame.
        :type df: pandas.DataFrame
        :param text_column: Text column to fall back to when tokens
            is absent.
        :type text_column: str
        :param label_column: Column containing class labels.
        :type label_column: str
        :param top_k: Number of n-grams shown per chart.
        :type top_k: int
        :returns: None
        :rtype: None
        """
        ngram_names = {1: "Unigrams", 2: "Bigrams", 3: "Trigrams"}

        for label in sorted(df[label_column].unique()):
            label_df = df[df[label_column] == label]
            for n in [1, 2, 3]:
                ngram_data = self.compute_label_top_ngrams(
                    label_df,
                    text_column=text_column,
                    label_column=label_column,
                    n=n,
                    top_k=top_k,
                )
                if not ngram_data.get(label):
                    continue

                ngrams_list, counts_list = zip(*ngram_data[label])
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = sns.color_palette(
                    "viridis", len(ngrams_list)
                )
                bars = ax.bar(
                    range(len(ngrams_list)), counts_list,
                    color=colors, edgecolor="black", alpha=0.8,
                )
                for bar in bars:
                    h = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2., h,
                        f"{int(h)}",
                        ha="center", va="bottom",
                        fontsize=10, weight="bold",
                    )
                ax.set_xticks(range(len(ngrams_list)))
                ax.set_xticklabels(
                    ngrams_list, rotation=45,
                    ha="right", fontsize=10,
                )
                ax.set_ylabel(
                    "Frequency", fontsize=12, weight="bold"
                )
                ax.set_xlabel(
                    "N-gram", fontsize=12, weight="bold"
                )
                ax.set_title(
                    f"Top {top_k} {ngram_names[n]}"
                    f" in {label} Statements",
                    fontsize=14, weight="bold", pad=20,
                )
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    self.output_dir
                    / f"{label.lower()}_{ngram_names[n].lower()}.png",
                    dpi=300, bbox_inches="tight",
                )
                print(f"  N-gram saved: {label} / n={n}")
                plt.close()

    def plot_emoji_frequency(self, df: pd.DataFrame) -> None:
        """
        2 × 2 bar chart of the top 10 emojis per label.

        :param df: Feature DataFrame with an emojis list column.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        labels = df["status"].unique()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # 2d to 1d matrix
        for idx, label in enumerate(labels):
            all_emojis: List[str] = []
            for lst in df[df["status"] == label]["emojis"]:
                if isinstance(lst, list):
                    all_emojis.extend(lst)
            if all_emojis:
                top = Counter(all_emojis).most_common(10)
                display, counts = [], []
                for em, cnt in top:
                    name = emoji.demojize(em).replace(":", "")
                    label_text = (
                        "Unknown"
                        if em == "□" or len(em) == 0
                        else f"{em}\n({name})"
                    )
                    display.append(label_text)
                    counts.append(cnt)
                bars = axes[idx].bar(
                    range(len(display)), counts,
                    color=self._colour(label),
                    edgecolor="black", alpha=0.8,
                )
                axes[idx].set_xticks(range(len(display)))
                axes[idx].set_xticklabels(
                    display, fontsize=9, rotation=45, ha="right"
                )
                axes[idx].set_ylabel(
                    "Frequency", fontsize=10, weight="bold"
                )
                axes[idx].set_title(
                    f"Emoji Frequency – {label}",
                    fontsize=12, weight="bold",
                )
                for bar in bars:
                    h = bar.get_height()
                    axes[idx].text(
                        bar.get_x() + bar.get_width() / 2., h,
                        f"{int(h)}", ha="center", va="bottom",
                        fontsize=9, weight="bold",
                    )
                axes[idx].grid(axis="y", alpha=0.3)
            else:
                axes[idx].text(
                    0.5, 0.5, "No emojis found",
                    ha="center", va="center", fontsize=12,
                )
                axes[idx].set_title(
                    f"Emoji Frequency – {label}",
                    fontsize=12, weight="bold",
                )
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "emoji_frequency.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def plot_emoji_presence_rate(self, df: pd.DataFrame) -> None:
        """
        Bar chart of the percentage of posts containing at least one
        emoji, per label.

        :param df: Feature DataFrame with an emoji_count column.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        rates: Dict[str, float] = {
            label: (
                (df[df["status"] == label]["emoji_count"] > 0).sum()
                / len(df[df["status"] == label])
            ) * 100
            for label in df["status"].unique()
        }
        labels = list(rates.keys())
        values = list(rates.values())
        bars = ax.bar(
            labels, values,
            color=[self._colour(lbl) for lbl in labels],
            edgecolor="none", alpha=0.8,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., h,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=11, weight="bold",
            )
        ax.set_ylabel(
            "Percentage of Posts (%)", fontsize=12, weight="bold"
        )
        ax.set_xlabel(
            "Mental Health Status", fontsize=12, weight="bold"
        )
        ax.set_title(
            "Emoji Presence Rate by Label",
            fontsize=14, weight="bold", pad=20,
        )
        ax.set_ylim(0, max(values) + 10 if values else 10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "emoji_presence_rate.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def plot_top_emoticons(self, df: pd.DataFrame) -> None:
        """
        2 × 2 horizontal bar chart of the top 10 emoticons per label.

        :param df: Feature DataFrame with an emoticons list column.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        labels = df["status"].unique()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for idx, label in enumerate(labels):
            all_emo: List[str] = []
            for lst in df[df["status"] == label]["emoticons"]:
                if isinstance(lst, list):
                    all_emo.extend(lst)
            if all_emo:
                top = Counter(all_emo).most_common(10)
                emoticons, counts = zip(*top)
                bars = axes[idx].barh(
                    range(len(emoticons)), counts,
                    color=self._colour(label),
                    edgecolor="none", alpha=0.8,
                )
                axes[idx].set_yticks(range(len(emoticons)))
                axes[idx].set_yticklabels(
                    list(emoticons), fontsize=10
                )
                axes[idx].set_xlabel(
                    "Frequency", fontsize=10, weight="bold"
                )
                axes[idx].set_title(
                    f"Emoticon Frequency – {label}",
                    fontsize=12, weight="bold",
                )
                for bar in bars:
                    w = bar.get_width()
                    axes[idx].text(
                        w + max(counts) * 0.01,
                        bar.get_y() + bar.get_height() / 2.,
                        f"{int(w)}", va="center", ha="left",
                        fontsize=9, weight="bold",
                    )
                axes[idx].invert_yaxis()
                axes[idx].grid(axis="x", alpha=0.3)
            else:
                axes[idx].text(
                    0.5, 0.5, "No emoticons found",
                    ha="center", va="center", fontsize=12,
                )
                axes[idx].set_title(
                    f"Emoticon Frequency – {label}",
                    fontsize=12, weight="bold",
                )
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "emoticon_frequency.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def plot_emoticon_presence_rate(
        self, df: pd.DataFrame
    ) -> None:
        """
        Bar chart of the percentage of posts containing at least one
        emoticon, per label.

        :param df: Feature DataFrame with an emoticon_count column.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        rates: Dict[str, float] = {
            label: (
                (
                    df[df["status"] == label]["emoticon_count"] > 0
                ).sum()
                / len(df[df["status"] == label])
            ) * 100
            for label in df["status"].unique()
        }
        labels = list(rates.keys())
        values = list(rates.values())
        bars = ax.bar(
            labels, values,
            color=[self._colour(lbl) for lbl in labels],
            edgecolor="none", alpha=0.8,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., h,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=11, weight="bold",
            )
        ax.set_ylabel(
            "Percentage of Posts (%)", fontsize=12, weight="bold"
        )
        ax.set_xlabel(
            "Mental Health Status", fontsize=12, weight="bold"
        )
        ax.set_title(
            "Emoticon Presence Rate by Label",
            fontsize=14, weight="bold", pad=20,
        )
        ax.set_ylim(0, max(values) + 10 if values else 10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "emoticon_presence_rate.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def plot_vocabulary_diversity(self, df: pd.DataFrame) -> None:
        """
        Violin plot of type-token ratio (TTR) per label.

        :param df: Feature DataFrame with ttr and status
            columns.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.violinplot(
            data=df, x="status", y="ttr",
            palette=self._PALETTE, ax=ax,
        )
        y_min, y_max = df["ttr"].min(), df["ttr"].max()
        y_range = y_max - y_min
        pad = 0.05 if y_range < 0.1 else y_range * 0.1
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(
            "Vocabulary Diversity (Type-Token Ratio) by Label",
            fontsize=14, weight="bold", pad=20,
        )
        ax.set_xlabel(
            "Mental Health Status", fontsize=12, weight="bold"
        )
        ax.set_ylabel(
            "Type-Token Ratio (TTR)", fontsize=12, weight="bold"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "vocabulary_diversity.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def plot_top_words_table(
        self, label_vocab: Dict[str, Set[str]]
    ) -> None:
        """
        Table comparing total, unique, and shared word counts per label.

        :param label_vocab: Output of :meth:compute_shared_vocabulary.
        :type label_vocab: Dict[str, Set[str]]
        :returns: None
        :rtype: None
        """
        labels_list = list(label_vocab.keys())
        if len(labels_list) < 2:
            return

        shared_all = label_vocab[labels_list[0]].copy()
        for lbl in labels_list[1:]:
            shared_all &= label_vocab[lbl]

        rows = []
        for lbl in labels_list:
            unique = label_vocab[lbl] - shared_all
            total = len(label_vocab[lbl])
            rows.append({
                "Label":        lbl,
                "Total Words":  total,
                "Unique Words": len(unique),
                "Shared Words": total - len(unique),
            })

        table_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            loc="center",
            colWidths=[0.3, 0.2, 0.2, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        for i in range(len(table_df.columns)):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(
                weight="bold", color="white"
            )
        for i, row in enumerate(table_df.values):
            lbl = row[0]
            table[(i + 1, 0)].set_facecolor(self._colour(lbl))
            table[(i + 1, 0)].set_text_props(
                weight="bold", color="white"
            )
        plt.title(
            "Vocabulary Statistics by Label",
            fontsize=14, weight="bold", pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "vocabulary_statistics_table.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    # ORCHESTRATION

    def run_plots(self, df: pd.DataFrame) -> None:
        """
        Run every visualisation in the given order.

        :param df: Feature DataFrame produced by
            :meth:extract_features.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        print("\nStarting visualisations...")

        print("[1/13] Label Distribution Pie Chart...")
        self.plot_label_distribution(df)

        print("[2/13] Text Length Histograms...")
        self.plot_text_length_distribution(df)

        print("[3/13] Individual Text Length Boxplots...")
        self.plot_individual_boxplots(
            df, column="text_length", ylabel="Text Length",
            filename_prefix="text_length_boxplot",
        )

        print("[4/13] Normalised Punctuation Table...")
        summary_df = self.compute_normalized_punctuation(df)
        self.plot_punctuation_normalized_table(summary_df)

        print("[5/13] Word Clouds...")
        self.plot_wordclouds(df)

        print("[6/13] Computing shared vocabulary...")
        label_vocab = self.compute_shared_vocabulary(df)

        print("[6.5/13] Venn Diagram...")
        self.plot_status_word_venn(label_vocab)

        print(
            "[7/13] N-gram Analysis "
            "(12 plots: 3 sizes × 4 labels)..."
        )
        self.plot_label_ngrams(
            df, "cleaned_text", "status", top_k=10
        )

        print("[8/13]  Emoji Frequency...")
        self.plot_emoji_frequency(df)

        print("[9/13]  Emoji Presence Rate...")
        self.plot_emoji_presence_rate(df)

        print("[10/13] Emoticon Frequency...")
        self.plot_top_emoticons(df)

        print("[11/13] Emoticon Presence Rate...")
        self.plot_emoticon_presence_rate(df)

        print("[12/13] Vocabulary Diversity Violin Plot...")
        self.plot_vocabulary_diversity(df)

        print("[13/13] Vocabulary Statistics Table...")
        self.plot_top_words_table(label_vocab)

        print(
            f"\nAll plots saved to: {self.output_dir}"
        )

    def save_feature_summary(self, df: pd.DataFrame) -> None:
        """
        Write a plain-text feature summary to
        feature_summary.txt.

        :param df: Feature DataFrame produced by
            :meth:extract_features.
        :type df: pandas.DataFrame
        :returns: None
        :rtype: None
        """
        path = self.output_dir / "feature_summary.txt"
        cols = [
            "text_length", "char_count", "punct_density",
            "question_count", "exclamation_count",
            "ellipsis_count", "emoji_count",
            "emoticon_count", "ttr",
        ]
        with open(path, "w") as fh:
            fh.write("=" * 70 + "\n")
            fh.write(
                "MENTAL HEALTH TEXT CLASSIFICATION"
                " – FEATURE SUMMARY\n"
            )
            fh.write("=" * 70 + "\n\n")
            fh.write("DATASET STATISTICS\n")
            fh.write("-" * 70 + "\n")
            fh.write(f"Total samples : {len(df)}\n")
            fh.write(
                f"Unique labels : {df['status'].nunique()}\n"
            )
            fh.write(
                f"Label distribution:\n"
                f"{df['status'].value_counts()}\n\n"
            )
            fh.write("FEATURE STATISTICS\n")
            fh.write("-" * 70 + "\n")
            for col in cols:
                fh.write(
                    f"\n{col.replace('_', ' ').title()}:\n"
                    f"{df[col].describe()}\n"
                )
            fh.write("\n\nPER-LABEL STATISTICS\n")
            fh.write("-" * 70 + "\n")
            for label in df["status"].unique():
                ld = df[df["status"] == label]
                fh.write(f"\n{label}:\n")
                fh.write(f"  Samples            : {len(ld)}\n")
                fh.write(
                    f"  Avg text length    : "
                    f"{ld['text_length'].mean():.2f}\n"
                )
                fh.write(
                    f"  Avg punct density  : "
                    f"{ld['punct_density'].mean():.4f}\n"
                )
                fh.write(
                    f"  Avg question count : "
                    f"{ld['question_count'].mean():.2f}\n"
                )
                fh.write(
                    f"  Avg exclamation    : "
                    f"{ld['exclamation_count'].mean():.2f}\n"
                )
                fh.write(
                    f"  Avg ellipsis       : "
                    f"{ld['ellipsis_count'].mean():.2f}\n"
                )
                fh.write(
                    f"  Avg emoji count    : "
                    f"{ld['emoji_count'].mean():.2f}\n"
                )
                fh.write(
                    f"  Avg emoticon count : "
                    f"{ld['emoticon_count'].mean():.2f}\n"
                )
                fh.write(
                    f"  Avg TTR            : "
                    f"{ld['ttr'].mean():.4f}\n"
                )
        print(f"Feature summary saved to: {path}")
