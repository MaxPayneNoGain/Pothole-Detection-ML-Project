"""
Trains a pothole vs plain road classifier on the merged dataset and saves
plots and metrics to figures/. Run prepare_data.py first.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

DATA_DIR = Path("data/merged")
FIG_DIR = Path("figures")

IMG_SIZE = 128
BATCH_SIZE = 16
SEED = 42
EPOCHS_BASE = 10
EPOCHS_AUG = 25
EPOCHS_TL = 12

# pothole is 0 and plain is 1, so the sigmoid output is P(plain)
CLASSES = ["pothole", "plain"]
LABEL_NAMES = ["Pothole", "Normal Road"]


def list_images(cls):
    return sorted(p for p in (DATA_DIR / cls).iterdir() if not p.name.startswith("."))


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype="float32") / 255.0


def load_dataset():
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        for path in list_images(cls):
            try:
                X.append(load_image(path))
            except OSError:
                print(f"skipping unreadable image: {path}")
                continue
            y.append(label)
    return np.stack(X), np.array(y)


def plot_samples(X, y):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Pothole (top) vs Normal Road (bottom)", fontsize=14, fontweight="bold")
    for row in range(len(CLASSES)):
        idx = np.flatnonzero(y == row)
        for ax, j in zip(axes[row], idx[::max(1, len(idx) // 5)]):
            ax.imshow(X[j])
            ax.set_title(LABEL_NAMES[row], fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "samples.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_balance(y):
    counts = np.bincount(y, minlength=len(CLASSES))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(LABEL_NAMES, counts, color=["#E53935", "#43A047"], width=0.45)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(c), ha="center", fontsize=12, fontweight="bold")
    ax.set_title("Class Balance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Images")
    ax.set_ylim(0, counts.max() * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_augmentation(augment, image):
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    for i in range(1, 6):
        out = augment(image[None], training=True)[0].numpy()
        axes[i].imshow(np.clip(out, 0, 1))
        axes[i].set_title(f"Aug {i}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Augmentation Examples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "augmentation_examples.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_curves(history, title, fname):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for i, metric in enumerate(["accuracy", "loss"]):
        ax[i].plot(history.history[metric], label="Train", color="#E53935")
        ax[i].plot(history.history[f"val_{metric}"], label="Val", color="#FF9800")
        ax[i].set_title(metric.title())
        ax[i].set_xlabel("Epoch")
        ax[i].legend()
        ax[i].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, result, fname):
    cm = confusion_matrix(y_true, result["preds"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(ax=ax, colorbar=False, cmap="Reds")
    ax.set_title(f"{result['name']} - Acc {result['accuracy']:.1%}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#E53935", "#FB8C00", "#1E88E5"]
    x = np.arange(len(metrics))
    w = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (result, color) in enumerate(zip(results, colors)):
        vals = [result[m] for m in metrics]
        pos = x + (i - (len(results) - 1) / 2) * w
        ax.bar(pos, vals, w, label=result["name"], color=color)
        for px, v in zip(pos, vals):
            ax.text(px, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [m.title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_misclassified(X, y, result):
    wrong = np.flatnonzero(result["preds"] != y)[:8]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    fig.suptitle(f"Misclassified - {result['name']}", fontsize=14, fontweight="bold")
    for ax, i in zip(axes.flat, wrong):
        pred = result["preds"][i]
        conf = result["probs"][i] if pred == 1 else 1 - result["probs"][i]
        ax.imshow(X[i])
        ax.set_title(f"True: {LABEL_NAMES[y[i]]}\nPred: {LABEL_NAMES[pred]} ({conf:.2f})", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "misclassified.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_metrics(results, best, y_train, y_test, weights):
    counts = np.bincount(np.concatenate([y_train, y_test]))
    summary = {
        "dataset": {
            "pothole": int(counts[0]),
            "plain": int(counts[1]),
            "train": len(y_train),
            "test": len(y_test),
            "class_weights": weights,
        },
        "models": [{k: v for k, v in r.items() if k not in ("probs", "preds")} for r in results],
        "best_model": best["name"],
    }
    with open(FIG_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


def evaluate(model, X, y, name):
    probs = model.predict(X, batch_size=BATCH_SIZE, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    # headline precision/recall/f1 are for potholes, the class we actually need to catch
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", pos_label=0,
                                                  zero_division=0)
    result = {
        "name": name,
        "accuracy": float((preds == y).mean()),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(roc_auc_score(y, probs)),
        "probs": probs,
        "preds": preds,
    }
    print(f"\n[{name}] acc={result['accuracy']:.4f} f1={f1:.4f} auc={result['auc']:.4f}")
    print(classification_report(y, preds, target_names=LABEL_NAMES, zero_division=0))
    return result


def build_baseline():
    return models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])


def build_augment():
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
    ], name="augment")


def build_augmented_cnn():
    return models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # random layers are no-ops at predict time, so test images go through untouched
        build_augment(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        # pooling instead of flatten drops the params from about 4.3M to about 100k, the baseline overfit
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])


def build_transfer():
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    # around 1k images isn't enough to retrain 2.3M params without wrecking the imagenet features
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = build_augment()(inputs)
    # mobilenetv2 was trained on pixels in [-1, 1], ours are in [0, 1]
    x = layers.Rescaling(2.0, offset=-1.0)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs)


def class_weights(y):
    n_pothole, n_plain = np.bincount(y)
    # the full imbalance ratio made training unstable, so sqrt is a softer push toward plain
    return {0: 1.0, 1: float(np.sqrt(n_pothole / n_plain))}


def main():
    tf.keras.utils.set_random_seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    print(f"loaded {len(y)} images: {np.bincount(y)[0]} pothole, {np.bincount(y)[1]} plain")
    plot_samples(X, y)
    plot_class_balance(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"train: {len(y_train)} | test: {len(y_test)}")
    results = []

    print("\n=== baseline cnn ===")
    model = build_baseline()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=EPOCHS_BASE, batch_size=BATCH_SIZE,
                        validation_split=0.2, verbose=2)
    plot_curves(history, "Baseline CNN", "baseline_curves.png")
    result = evaluate(model, X_test, y_test, "Baseline CNN")
    plot_confusion(y_test, result, "baseline_confusion.png")
    results.append(result)

    weights = class_weights(y_train)
    print(f"class weights: {weights}")
    plot_augmentation(build_augment(), X_train[0])

    print("\n=== cnn + augmentation ===")
    model = build_augmented_cnn()
    model.compile(optimizer=optimizers.Adam(5e-4), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=EPOCHS_AUG, batch_size=BATCH_SIZE,
                        validation_split=0.2, class_weight=weights,
                        callbacks=[early_stop], verbose=2)
    plot_curves(history, "CNN + Augmentation", "aug_curves.png")
    result = evaluate(model, X_test, y_test, "CNN + Augmentation")
    plot_confusion(y_test, result, "aug_confusion.png")
    results.append(result)

    print("\n=== mobilenetv2 transfer ===")
    model = build_transfer()
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=EPOCHS_TL, batch_size=BATCH_SIZE,
                        validation_split=0.2, class_weight=weights,
                        callbacks=[early_stop], verbose=2)
    plot_curves(history, "MobileNetV2 Transfer", "mobilenetv2_curves.png")
    result = evaluate(model, X_test, y_test, "MobileNetV2 Transfer")
    plot_confusion(y_test, result, "mobilenetv2_confusion.png")
    results.append(result)

    plot_comparison(results)
    best = max(results, key=lambda r: r["f1"])
    plot_misclassified(X_test, y_test, best)
    save_metrics(results, best, y_train, y_test, weights)

    print()
    for r in results:
        print(f"{r['name']:22s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}")
    print(f"best by f1: {best['name']}")


if __name__ == "__main__":
    main()
