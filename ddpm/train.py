import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_lr(model, train_loader, start_lr=1e-7, end_lr=1, num_iter=100,use_time=True):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    loss_fn = nn.MSELoss()
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    lr_mult = (end_lr / start_lr) ** (1 / max(1, num_iter - 1))
    lr = start_lr
    lrs, losses = [], []
    data_iter = iter(train_loader)

    for _ in tqdm(range(num_iter), desc='LR finder'):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x_noisy, noise = batch[0].to(device), batch[1].to(device)
        t = batch[2].to(device) if use_time  else None

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        noise_pred = model(x_noisy, t)
        loss = loss_fn(noise_pred, noise)

        if not torch.isfinite(loss):
            print(f'Stopping early: non-finite loss at lr={lr:.2e}')
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())
        lr *= lr_mult

    model.load_state_dict(initial_state)

    if len(losses) == 0:
        raise RuntimeError("LR finder failed before recording any losses.")

    plt.figure(figsize=(7, 4))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('MSE loss')
    plt.title('LR finder')
    plt.grid(True)
    plt.show()

    best_idx = losses.index(min(losses))
    suggested_lr = lrs[max(0, best_idx - 1)]
    return suggested_lr


def train(model, train_loader, test_loader, epochs=100, lr=1e-2, weight_decay=1e-4,
          early_stopping_patience=10, save_path='model.pkl', writer=None, use_time=True):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        epoch_train_loss = 0
        for batch in tqdm(train_loader, leave=False, desc='train'):
            x_noisy, noise, t = (batch[0].to(device), batch[1].to(device),
                                 batch[2].to(device) if use_time else None,)

            with torch.amp.autocast('cuda'):
                noise_pred = model(x_noisy, t)
                loss = loss_fn(noise_pred, noise)

            epoch_train_loss += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False, desc='test'):
                x_noisy, noise, t = (batch[0].to(device), batch[1].to(device),
                                 batch[2].to(device) if use_time else None,)
                with torch.amp.autocast('cuda'):
                    noise_pred = model(x_noisy,t)
                    epoch_test_loss += loss_fn(noise_pred, noise).item()
            test_loss = epoch_test_loss / len(test_loader)
            lr_scheduler.step(test_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch} | train loss: {train_losses[-1]:.4f} | test loss: {test_losses[-1]:.4f}")

        if writer:
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Params/LR', optimizer.param_groups[0]['lr'], epoch)

        if test_losses[-1] < best_test_loss:
            best_test_loss = test_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(save_path))
    return train_losses, test_losses
