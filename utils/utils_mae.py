import os
import torch
from tqdm import tqdm
from utils.utils import get_lr


def mae_one_epoch(model_train, model, optimizer, epoch, warmup_scheduler, warmup_epoch,
                  epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, loss_fuc, fp16, scaler, save_period, save_dir, local_rank, lr_scheduler):
    total_loss, val_loss = 0, 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        if epoch < warmup_epoch:
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()[0]

        imgs, labels = batch
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            labels = torch.from_numpy(labels).type(torch.FloatTensor)
            if cuda:
                imgs = imgs.cuda(local_rank)
                labels = labels.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            loss = loss_fuc(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast(enabled=scaler is not None):
                outputs = model_train(imgs)
                loss = loss_fuc(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': warm_lr if epoch < warmup_epoch else get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, labels = batch
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            labels = torch.from_numpy(labels).type(torch.FloatTensor)
            if cuda:
                imgs = imgs.cuda(local_rank)
                labels = labels.cuda(local_rank)

            outputs = model_train(imgs)
            loss = loss_fuc(outputs, labels)

            val_loss += loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'lr': warm_lr if epoch < warmup_epoch else get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        save_file = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'lr_scheduler': lr_scheduler.state_dict(),
                     'epoch': epoch}
        if fp16:
            save_file["scaler"] = scaler.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'exp-%03d-train_loss%.3f-val_loss%.3f.pth' % (
            (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        torch.save(save_file, os.path.join(save_dir, "last_weights.pth"))
