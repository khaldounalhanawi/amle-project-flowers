"""Recompute every number shown in the presentation and cache it in results/.

Run once with `uv run python analysis.py`; the presentation only reads the cache.
"""
import json
from collections import Counter
from pathlib import Path

import keras
import numpy as np
from keras.applications import mobilenet_v2
from PIL import Image

IMAGE_SIZE = (160, 160)
CLASSES = ["rose", "sunflower", "tulip"]
DATA = Path("data")
OUT = Path("presentation_files/results")
OUT.mkdir(exist_ok=True)


def load(split, shuffle=False):
    return keras.utils.image_dataset_from_directory(
        DATA / split, image_size=IMAGE_SIZE, batch_size=32, label_mode="int", shuffle=shuffle, seed=42
    )


def base_model(top):
    return mobilenet_v2.MobileNetV2(
        include_top=top, weights="imagenet", input_shape=IMAGE_SIZE + (3,), pooling="avg"
    )


def features(base, ds):
    xs, ys = [], []
    for images, labels in ds:
        xs.append(base.predict(mobilenet_v2.preprocess_input(images), verbose=0))
        ys.append(labels.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def main():
    keras.utils.set_random_seed(42)
    train_s = load("train", shuffle=False)
    train, valid, test = load("train", shuffle=True), load("validation"), load("test")
    results = {"classes": CLASSES}

    # 1. Image sizes of every photo
    results["widths"] = [Image.open(p).size[0] for p in sorted(DATA.glob("*/*/*.jpg"))]

    # 2. What the original ImageNet classifier says about our training photos
    full = base_model(True)
    counts = {c: Counter() for c in CLASSES}
    for images, labels in train_s:
        probs = full.predict(mobilenet_v2.preprocess_input(images), verbose=0)
        for lab, row in zip(labels.numpy(), mobilenet_v2.decode_predictions(probs, top=1)):
            counts[CLASSES[lab]][row[0][1]] += 1
    results["imagenet_top1"] = {c: counts[c].most_common(5) for c in CLASSES}

    # 3. Dense head on frozen features
    base2 = base_model(False)
    base2.trainable = False
    X_train, y_train = features(base2, train_s)
    X_valid, y_valid = features(base2, valid)
    X_test, y_test = features(base2, test)

    keras.utils.clear_session()
    keras.utils.set_random_seed(42)
    head = keras.Sequential([
        keras.Input(shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation="softmax", kernel_regularizer=keras.regularizers.L2(0.1)),
    ])
    head.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    h = head.fit(X_train, y_train, epochs=30, verbose=0, validation_data=(X_valid, y_valid))
    dense = {
        "history": {k: [float(v) for v in vals] for k, vals in h.history.items()},
        "val_acc": float(head.evaluate(X_valid, y_valid, verbose=0)[1]),
        "test_acc": float(head.evaluate(X_test, y_test, verbose=0)[1]),
        "test_pred": head.predict(X_test, verbose=0).argmax(1).tolist(),
    }

    # 4. Augmented end-to-end model
    keras.utils.clear_session()
    keras.utils.set_random_seed(42)
    base3 = base_model(False)
    base3.trainable = False
    augmentation = keras.Sequential(
        [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.1), keras.layers.RandomZoom(0.1)]
    )
    model = keras.Sequential([
        keras.Input(shape=IMAGE_SIZE + (3,)),
        augmentation,
        keras.layers.Rescaling(scale=1 / 127.5, offset=-1),
        base3,
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(1e-4), metrics=["accuracy"])
    h = model.fit(train, validation_data=valid, epochs=30, verbose=0)
    aug = {
        "history": {k: [float(v) for v in vals] for k, vals in h.history.items()},
        "val_acc": float(model.evaluate(valid, verbose=0)[1]),
        "test_acc": float(model.evaluate(test, verbose=0)[1]),
        "test_pred": model.predict(test, verbose=0).argmax(1).tolist(),
    }

    results["dense_head"], results["augmented"] = dense, aug
    results["test_true"] = y_test.tolist()
    results["test_files"] = [str(p) for p in test.file_paths]
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print({k: (v["val_acc"], v["test_acc"]) for k, v in [("dense", dense), ("aug", aug)]})


if __name__ == "__main__":
    main()
