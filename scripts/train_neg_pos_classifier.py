from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class NegPosDataset(Dataset):
    def __init__(
        self,
        embed_dir: Path,
        transform=None,
    ):
        self.embed_files = list(embed_dir.glob("*.npz"))
        self.transform = transform

    def __len__(self):
        return len(self.embed_files)

    def __getitem__(self, ind):
        data = np.load(self.embed_files[ind])["feat"]
        data = np.ascontiguousarray(data)  # N x 196 x 1024
        data = torch.from_numpy(data)
        if self.transform:
            data = self.transform(data)

        label = torch.tensor(
            1 if "pos" in self.embed_files[ind].name else 0, dtype=torch.long
        )
        return data, label


class LinearProbe(nn.Module):
    def __init__(self, c: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(c, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        return self.proj(x)  # (B, L, K)


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims=(128, 64), out_dim: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x = x.mean(dim=1)  # (B, C)  pool over tokens/patches
        return self.net(x)  # (B, K)


def train_one_epoch(loader, model, criterion, optimizer, device, print_batches=True):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        total_correct += correct

        if print_batches:
            batch_acc = 100.0 * correct / batch_size
            print(f"Train Loss: {loss.item():.4f}, Accuracy: {batch_acc:.2f}%")

    avg_loss = total_loss / total_examples
    avg_acc = 100.0 * total_correct / total_examples
    return avg_loss, avg_acc


def eval_model(loader, model, criterion, device, print_batches=True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            total_correct += correct

            if print_batches:
                batch_acc = 100.0 * correct / batch_size
                print(f"Eval Loss: {loss.item():.4f}, Accuracy: {batch_acc:.2f}%")

    avg_loss = total_loss / total_examples
    avg_acc = 100.0 * total_correct / total_examples
    return avg_loss, avg_acc


def write_info_in_tensorboard(writer, epoch, loss, accuracy, stage):
    loss_scalar_dict = dict()
    loss_scalar_dict[stage] = loss
    accuracy_scalar_dict = dict()
    accuracy_scalar_dict[stage] = accuracy
    writer.add_scalars("loss", loss_scalar_dict, epoch)
    writer.add_scalars("accuracy", accuracy_scalar_dict, epoch)


torch.manual_seed(0)
cfg = dict(
    num_workers=1,
    batch_size=32,
    no_epochs=50,
    exp="neg_pos_classifier_test",
    save_dir=Path("/home/fatemeh/Downloads/hedg/results/training"),
    data_dir=Path("/home/fatemeh/Downloads/hedg/results/test_dataset"),
)
cfg = OmegaConf.create(cfg)

cfg.save_dir.mkdir(parents=True, exist_ok=True)
cfg.tensorboard_file = cfg.save_dir / f"tensorboard/{cfg.exp}"
cfg.tensorboard_file.mkdir(parents=True, exist_ok=True)

embed_dir = cfg.data_dir / "embeddings"
all_dataset = NegPosDataset(embed_dir=embed_dir)
lengths = [int(0.8 * len(all_dataset)), len(all_dataset) - int(0.8 * len(all_dataset))]
train_dataset, eval_dataset = torch.utils.data.random_split(all_dataset, lengths)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    # sampler=sampler,
    shuffle=True,
    num_workers=cfg.num_workers,
    drop_last=False,
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=len(eval_dataset),
    shuffle=False,
    num_workers=cfg.num_workers,
    drop_last=False,
)

model = MLP(in_dim=1024, hidden_dims=(64, 32), out_dim=2, dropout=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_accuracy = 0.0
with tensorboard.SummaryWriter(cfg.tensorboard_file) as writer:
    for epoch in tqdm(range(1, cfg.no_epochs + 1)):
        start_time = datetime.now().replace(microsecond=0)
        train_loss, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, device
        )
        end_time_train = datetime.now().replace(microsecond=0)
        eval_loss, eval_acc = eval_model(eval_loader, model, criterion, device)
        end_time_eval = datetime.now().replace(microsecond=0)
        print(f"Epoch {epoch:02d} time taken: {end_time_eval - start_time}")
        print(
            f"{end_time_train}, epoch {epoch:02d}/{cfg.no_epochs} loss={train_loss:.4f} eval_loss={eval_loss:.4f}"
        )
        print(
            f"{end_time_eval}, epoch {epoch:02d}/{cfg.no_epochs} acc={train_acc:.2f}% eval_acc={eval_acc:.2f}%"
        )
        write_info_in_tensorboard(writer, epoch, train_loss, train_acc, "train")
        write_info_in_tensorboard(writer, epoch, eval_loss, eval_acc, "eval")

        # Save best model
        if eval_acc > best_accuracy:
            best_accuracy = eval_acc
            torch.save(
                {"model": model.state_dict(), "epoch": epoch},
                cfg.save_dir / f"best_{np.exp}.pt",
            )
            print(f"Best model accuracy: {best_accuracy:.2f}% at epoch: {epoch}")
