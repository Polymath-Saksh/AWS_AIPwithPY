# Udacity AWS AI Programming with Python Nanodegree

## Nanodegree Final Project: Image Classifier

### Training an image classifier

Train a new network on a dataset with `train.py`.

**Basic usage:**

```bash
python train.py data_directory
```

Prints out training loss, validation loss, and validation accuracy as the network trains.

**Options:**

- Set directory to save checkpoints:

    ```bash
    python train.py data_dir --save_dir save_directory
    ```

- Choose architecture:

    ```bash
    python train.py data_dir --arch "vgg13"
    ```

- Set hyperparameters:

    ```bash
    python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    ```

- Use GPU for training:

    ```bash
    python train.py data_dir --gpu
    ```

### Predicting the class from an image using a saved model

Predict flower name from an image with `predict.py` along with the probability of that name.

**Basic usage:**

```bash
python predict.py /path/to/image checkpoint
```

**Options:**

- Return top K most likely classes:

    ```bash
    python predict.py input checkpoint --top_k 3
    ```

- Use a mapping of categories to real names:

    ```bash
    python predict.py input checkpoint --category_names cat_to_name.json
    ```

- Use GPU for inference:

    ```bash
    python predict.py input checkpoint --gpu
    ```
