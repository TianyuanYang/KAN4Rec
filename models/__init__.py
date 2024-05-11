from .kan4rec import KAN4RecModel

MODELS = {
    KAN4RecModel.code(): KAN4RecModel,
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
