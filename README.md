# Flower Classification Project

A one-day project with 390 photos of roses, sunflowers and tulips. You classify the photos with only 100 training photos per flower: use a pretrained network with a dense head, improve it with data augmentation, and find out which photos the best model gets wrong. At the end, you try your model on photos of your own in a small Streamlit app and deploy it on Streamlit Community Cloud.

## Learning Objectives

By the end of this project, you should be able to:

- Load training, validation and test images from folders, and keep the test set locked until the final evaluation.
- Train a frozen pretrained base with a dense head, improve the model with data augmentation, and choose the best model by its validation accuracy.
- Compare every model in one table of validation and test accuracies, and use a confusion matrix and the misclassified test photos to identify where the chosen model makes its mistakes.

## Learning Path

| File | Description |
|---|---|
| [**01 - Flower Classification Project**](01_flower_classification_project.ipynb) | The project: the business problem, the data description, the task in seven steps, the last one in the Streamlit app. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**app.py**](app.py) | A Streamlit app that loads a saved model from `models/` and classifies an uploaded photo (step 7). |
| [**Assets**](assets/) | The diagram of the project's steps shown in the notebook, with its Mermaid source in `assets/diagrams/`. |
| [**Data**](data/) | The photos, one folder per flower: 100 per class in `train/`, 20 per class in `validation/` and 10 per class in `test/`, with the photographers in `LICENSE.txt`. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd my-flower-project`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies, including TensorFlow, Keras and Streamlit, and creates a virtual environment in `.venv/`. TensorFlow is a large package, so the first install takes a few minutes.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

---

### 6. Run the App

After you have saved a model to `models/` in step 5 of the notebook, start the app from the project root folder:

```bash
uv run streamlit run app.py
```

It opens in your browser. Choose a model and upload a photo of a rose, a sunflower or a tulip.

> [!NOTE]
> The app passes the raw photo, with pixel values from 0 to 255, to the model. It works with models that include their preprocessing, such as `preprocess_input`, as a step inside the model.

---

### 7. Deploy the App to Streamlit Community Cloud

Once the app works on your machine, deploy it on [Streamlit Community Cloud](https://streamlit.io/cloud) so that anyone can try your model in the browser, without installing anything.

1. **Commit your best model.** The `models/` folder is ignored by Git, so add the model file explicitly, then commit and push it:

   ```bash
   git add -f models/<model-name>.keras
   git commit -m "Add the trained flower model"
   git push
   ```

2. **Sign in.** Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. **Create the app.** Click **Create app**, then **Yup, I have an app**, and fill in:
   - **Repository**: `<your-username>/<repo-name>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): a subdomain of your choice, for example `<your-name>-flowers.streamlit.app`
4. **Choose the Python version.** Click **Advanced settings**, select Python **3.13** to match `.python-version`, and click **Save**.
5. **Deploy.** Click **Deploy**. Community Cloud installs the dependencies from `uv.lock`; TensorFlow is large, so the first build takes a few minutes.
6. **Share the link.** Open the app, upload a flower photo and share the URL with your team.

> [!IMPORTANT]
> To deploy, you need **admin** rights on the repository, so in pairs or groups the person who created the repository deploys the app.

> [!NOTE]
> An app deployed from a private repository is private by default. Community Cloud allows only **one private app** per account, and you need to grant it access to your private repositories in **Settings → Linked accounts**. To share the app, invite viewers by email under **Share**, or make it public.

## Data and Credits

The photos are part of the repository, in `data/` (30 MB). The pretrained weights are downloaded the first time you load a network from `keras.applications` and stored in `~/.keras/models`: 9 MB for MobileNetV2 without its top, loaded from disk on later runs.

- **Flower photos**: 390 photos from the [TensorFlow flower photos](https://www.tensorflow.org/datasets/catalog/tf_flowers) collection, taken from Flickr. Every photo is licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/); its photographer is listed in [`data/LICENSE.txt`](data/LICENSE.txt).

## References & Further Reading

- [**Keras: Transfer learning and fine-tuning**](https://keras.io/guides/transfer_learning/): Freezing a base, training a head and fine-tuning, including the `BatchNormalization` layers.
- [**Keras: Image classification from scratch**](https://keras.io/examples/vision/image_classification_from_scratch/): Loading images from folders and data augmentation.
- [**Keras Applications**](https://keras.io/api/applications/): The pretrained networks available in Keras, with their size, accuracy and `preprocess_input`.
- [**TensorFlow: Transfer learning with a pretrained ConvNet**](https://www.tensorflow.org/tutorials/images/transfer_learning): MobileNetV2 as a frozen base and fine-tuning, step by step.
- [**Streamlit documentation**](https://docs.streamlit.io/): Building and running small web apps in Python, including `st.file_uploader` and `st.cache_resource`.
- [**Streamlit Community Cloud: Deploy your app**](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy): Deploying an app from a GitHub repository, choosing the Python version and sharing the app.
- [**Sandler et al. (2018), MobileNetV2: Inverted Residuals and Linear Bottlenecks**](https://arxiv.org/abs/1801.04381): The paper behind MobileNetV2.
