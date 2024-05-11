from .kan4rec import KAN4RecTrainer
TRAINERS = {
    KAN4RecTrainer.code(): KAN4RecTrainer,
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
