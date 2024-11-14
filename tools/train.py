from core.models import *
from tools.utils import *
import random
from tqdm import tqdm


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)

    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    parser.add_argument('--checkpoints', type=str, default='')
    return parser


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)
    best_acc = 0
    stop = 0
    if os.path.exists(args.checkpoints):
        pass
    else:
        os.makedirs(args.checkpoints)
    write_log(args.checkpoints, [str(args), '\n'])
    for e in range(1, args.n_epoch + 1):
        model.train()
        train_loss_clf = AverageMeter()
        train_loss_transfer = AverageMeter()
        train_loss_contrastive = AverageMeter()
        train_loss_total = AverageMeter()
        model.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        loop = tqdm(range(n_batch), total=n_batch)
        for iteration in loop:
            data_source, label_source, asc_source = next(iter_source)  # .next()
            data_target, _, asc_target = next(iter_target)  # .next()
            data_source, label_source, asc_source = data_source.to(args.device), label_source.to(args.device), asc_source.to(args.device)
            data_target, asc_target = data_target.to(args.device), asc_target.to(args.device)

            clf_loss, transfer_loss, contrastive_loss = model(data_source, data_target, label_source, asc_source, asc_target)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss + 0.01 * contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_contrastive.update(contrastive_loss.item())
            train_loss_total.update(loss.item())
            loop.set_description(f'Epoch [{e}/{args.n_epoch}]')
            info_dct={'lr': optimizer.state_dict()["param_groups"][1]["lr"],
                      'clf_loss': train_loss_clf.avg,
                      'transfer_loss': train_loss_transfer.avg,
                      'contrastive_loss': train_loss_contrastive.avg,
                      'total_loss': train_loss_total.avg}
            loop.set_postfix(info_dct)
            if ((iteration + 1) % 100) == 0:
                iteration_info = 'Iteration: [{:2d}/{}], lr: {:.4f}, cls_loss: {:.4f}, ' \
                                 'transfer_loss: {:.4f}, contrastive_loss: {:.4f}, total_loss: {:.4f}\n'.format(iteration, n_batch,
                                  optimizer.state_dict()["param_groups"][1]["lr"], train_loss_clf.avg,
                    train_loss_transfer.avg, train_loss_contrastive.avg, train_loss_total.avg)
                write_log(args.checkpoints, [iteration_info, ])

        epoch_info = 'Epoch: [{:2d}/{}], lr: {:.4f}, cls_loss: {:.4f}, transfer_loss: {:.4f}, contrastive_loss: {:.4f}, total_loss: {:.4f}'.format(
            e, args.n_epoch, optimizer.state_dict()["param_groups"][1]["lr"], train_loss_clf.avg,
            train_loss_transfer.avg, train_loss_contrastive.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, args)
        epoch_info += ', test_loss: {:4f}, test_acc: {:.4f}\n'.format(test_loss, test_acc)
        write_log(args.checkpoints, [epoch_info, ])
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.checkpoints + "/best_model.pth")
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(epoch_info)
            break
    done_info = 'Transfer result: {:.4f}\n'.format(best_acc)
    print(done_info)
    write_log(args.checkpoints, [done_info, ])


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    setattr(args, "seed", (args.seed if args.seed != 0 else random.randint(1, 10000000)))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)

    model_state_dict = frequency_select(source_loader, target_train_loader, args)
    model = TFSNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
        use_bottleneck=args.use_bottleneck).to(args.device)
    model.load_state_dict(model_state_dict, strict=False)

    optimizer = get_optimizer(model, args)
    if args.lr_scheduler:
        adjust_epoch = [int(args.max_iter * (2 / 3)), int(args.max_iter * (5 / 6))]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, adjust_epoch, gamma=0.1, last_epoch=-1)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)


if __name__ == "__main__":
    main()
