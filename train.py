import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import DeepfakeDataset
from config import *


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("        DEEPFAKE DETECTION TRAINING")
    print("=" * 60)

    print(f"Device : {DEVICE}")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Epochs : {EPOCHS}")
    print()


    # ======================================================
    # Collect image paths
    # ======================================================

    image_paths = []
    labels = []

    folders = [

        (TRAIN_REAL_DIR, 0),
        (TRAIN_FAKE_DIR, 1),

        (VAL_REAL_DIR, 0),
        (VAL_FAKE_DIR, 1)

    ]

    for folder, label in folders:

        if not os.path.exists(folder):

            print(
                f"[WARNING] Folder not found: {folder}"
            )

            continue

        for file in os.listdir(folder):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                image_paths.append(
                    os.path.join(folder, file)
                )

                labels.append(label)


    # ======================================================
    # Dataset information
    # ======================================================

    print(
        f"Total images : {len(image_paths)}"
    )

    real_count = labels.count(0)
    fake_count = labels.count(1)

    print(
        f"Real images  : {real_count}"
    )

    print(
        f"Fake images  : {fake_count}"
    )

    print()


    if len(image_paths) == 0:

        print("ERROR: No images found.")
        return


    # ======================================================
    # Train / validation split
    # ======================================================

    train_images, val_images, train_labels, val_labels = train_test_split(

        image_paths,
        labels,

        test_size=0.20,

        random_state=RANDOM_SEED,

        stratify=labels

    )


    print(
        f"Training images   : {len(train_images)}"
    )

    print(
        f"Validation images : {len(val_images)}"
    )

    print()


    # ======================================================
    # ImageNet normalization
    # ======================================================

    imagenet_mean = [
        0.485,
        0.456,
        0.406
    ]

    imagenet_std = [
        0.229,
        0.224,
        0.225
    ]


    # ======================================================
    # Training transforms
    # ======================================================

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            10
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        )

    ])


    # ======================================================
    # Validation transforms
    # ======================================================

    val_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        )

    ])


    # ======================================================
    # Dataset
    # ======================================================

    train_dataset = DeepfakeDataset(

        train_images,

        train_labels,

        train_transform

    )


    val_dataset = DeepfakeDataset(

        val_images,

        val_labels,

        val_transform

    )


    # ======================================================
    # DataLoader
    # ======================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )


    # ======================================================
    # Model
    # ======================================================

    print("Loading EfficientNet-B0...")

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )


    # Replace classifier

    model.classifier[1] = nn.Linear(

        model.classifier[1].in_features,

        NUM_CLASSES

    )


    model = model.to(DEVICE)


    # ======================================================
    # Class weights
    # ======================================================

    train_real = train_labels.count(0)
    train_fake = train_labels.count(1)

    total_train = train_real + train_fake


    real_weight = (
        total_train /
        (2 * train_real)
    )

    fake_weight = (
        total_train /
        (2 * train_fake)
    )


    class_weights = torch.tensor(

        [
            real_weight,
            fake_weight
        ],

        dtype=torch.float32

    ).to(DEVICE)


    print()
    print("Class weights:")

    print(
        f"REAL : {real_weight:.4f}"
    )

    print(
        f"FAKE : {fake_weight:.4f}"
    )

    print()


    # ======================================================
    # Loss
    # ======================================================

    criterion = nn.CrossEntropyLoss(

        weight=class_weights

    )


    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    # ======================================================
    # Learning-rate scheduler
    # ======================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2

    )


    # ======================================================
    # Training variables
    # ======================================================

    best_accuracy = 0.0

    epochs_without_improvement = 0


    # ======================================================
    # TRAINING LOOP
    # ======================================================

    for epoch in range(EPOCHS):

        print()
        print("=" * 60)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("=" * 60)


        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels_batch in train_loader:

            images = images.to(DEVICE)

            labels_batch = labels_batch.to(DEVICE)


            optimizer.zero_grad()


            outputs = model(images)


            loss = criterion(
                outputs,
                labels_batch
            )


            loss.backward()


            optimizer.step()


            running_loss += loss.item()


            _, predicted = outputs.max(1)


            total += labels_batch.size(0)


            correct += (
                predicted == labels_batch
            ).sum().item()


        train_accuracy = (
            100.0 * correct / total
        )


        average_loss = (
            running_loss /
            len(train_loader)
        )


        print(
            f"Loss       : {average_loss:.4f}"
        )

        print(
            f"Train Acc  : {train_accuracy:.2f}%"
        )


        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        correct = 0

        total = 0

        real_correct = 0
        real_total = 0

        fake_correct = 0
        fake_total = 0


        with torch.no_grad():

            for images, labels_batch in val_loader:

                images = images.to(DEVICE)

                labels_batch = labels_batch.to(DEVICE)


                outputs = model(images)


                _, predicted = outputs.max(1)


                total += labels_batch.size(0)


                correct += (
                    predicted == labels_batch
                ).sum().item()


                # REAL statistics

                real_mask = (
                    labels_batch == 0
                )

                real_total += (
                    real_mask.sum().item()
                )

                real_correct += (
                    (
                        predicted[real_mask] == 0
                    ).sum().item()
                )


                # FAKE statistics

                fake_mask = (
                    labels_batch == 1
                )

                fake_total += (
                    fake_mask.sum().item()
                )

                fake_correct += (
                    (
                        predicted[fake_mask] == 1
                    ).sum().item()
                )


        validation_accuracy = (
            100.0 * correct / total
        )


        real_accuracy = (
            100.0 * real_correct / real_total
            if real_total > 0
            else 0
        )


        fake_accuracy = (
            100.0 * fake_correct / fake_total
            if fake_total > 0
            else 0
        )


        print()

        print(
            f"Validation Acc : "
            f"{validation_accuracy:.2f}%"
        )

        print(
            f"REAL Accuracy   : "
            f"{real_accuracy:.2f}%"
        )

        print(
            f"FAKE Accuracy   : "
            f"{fake_accuracy:.2f}%"
        )


        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        scheduler.step(
            validation_accuracy
        )


        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning Rate   : {current_lr:.7f}"
        )


        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            epochs_without_improvement = 0


            torch.save(

                model.state_dict(),

                BEST_MODEL_PATH

            )


            print()
            print(
                "✓ Best model saved."
            )

            print(
                f"  Accuracy : "
                f"{best_accuracy:.2f}%"
            )


        else:

            epochs_without_improvement += 1


        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break


    # ======================================================
    # Save final model
    # ======================================================

    torch.save(

        model.state_dict(),

        LAST_MODEL_PATH

    )


    # ======================================================
    # Finished
    # ======================================================

    print()
    print("=" * 60)
    print("          TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best Validation Accuracy : "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Best model : "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Last model : "
        f"{LAST_MODEL_PATH}"
    )

    print("=" * 60)


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()