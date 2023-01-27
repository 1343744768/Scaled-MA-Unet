import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, warmup_scheduler, warmup_epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank, lr_scheduler):
    total_loss, total_f_score, val_loss, val_f_score = 0, 0, 0, 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        if epoch < warmup_epoch:
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()[0]

        imgs, pngs, labels = batch
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            pngs = torch.from_numpy(pngs).long()
            labels = torch.from_numpy(labels).type(torch.FloatTensor)
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast(enabled=scaler is not None):
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score': total_f_score / (iteration + 1),
                                'lr': warm_lr if epoch<warmup_epoch else get_lr(optimizer)})
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
        imgs, pngs, labels = batch
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            pngs = torch.from_numpy(pngs).long()
            labels = torch.from_numpy(labels).type(torch.FloatTensor)
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': warm_lr if epoch<warmup_epoch else get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(val_f_score / epoch_step_val, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        save_file = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'lr_scheduler': lr_scheduler.state_dict(),
                     'epoch': epoch}
        if fp16:
            save_file["scaler"] = scaler.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'exp-%03d-train_loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if eval_callback.period != 1:  # 如果不每轮验证，保存最好模型以val f1得分最大为标准
            if (val_f_score / epoch_step_val) > 0.5 and (val_f_score / epoch_step_val) >= max(loss_history.f_score):
                print('Save best model to best_epoch_weights.pth')
                torch.save(model.state_dict(), os.path.join(save_dir, "best_weights.pth"))
        else:   # 如果每轮验证，则保存最好模型以miou最大为标准
            if eval_callback.miou == max(eval_callback.mious):
                print('Save best model to best_epoch_weights.pth')
                torch.save(model.state_dict(), os.path.join(save_dir, "best_weights.pth"))
        torch.save(save_file, os.path.join(save_dir, "last_weights.pth"))
