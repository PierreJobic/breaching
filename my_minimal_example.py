import logging

import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader

import breaching

from breaching.cases.models import custom_models  # , model_preparation

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class data_cfg_default:
    modality = "vision"
    classes = 10
    shape = (3, 32, 32)
    normalize = True
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)


transforms = torchvision.transforms.Compose(
    [
        # torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=data_cfg_default.mean, std=data_cfg_default.std),
    ]
)

target_transform = torchvision.transforms.Compose(
    [
        # torchvision.transforms.Lambda(lambd=lambda x: F.one_hot(torch.tensor(x), num_classes=10)),
    ]
)


def main(batch_size=4):
    setup = dict(device=DEVICE, dtype=torch.float)

    # This could be your model:
    model = custom_models.McMahan_32_32()
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    # And your dataset:
    torch.manual_seed(908)
    dataset = torchvision.datasets.CIFAR10(root="~/data", train=False, transform=transforms, target_transform=target_transform)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data, labels = next(iter(trainloader))
    data_cfg_default.size = len(dataset)

    # This is the attacker:
    cfg_attack = breaching.get_attack_config("PJ_DLG")
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)
    print(attacker)

    # ## Simulate an attacked FL protocol
    # Server-side computation:
    server_payload = [
        dict(
            parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=data_cfg_default
        )
    ]

    # User-side computation:
    loss = loss_fn(model(data), labels)
    shared_data = [
        dict(
            gradients=torch.autograd.grad(loss, model.parameters()),
            buffers=None,
            metadata=dict(num_data_points=data.shape[0], labels=labels, local_hyperparams=None,),
        )
    ]

    # Attack:
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)

    # Do some processing of your choice here. Maybe save the output image?
    return attacker, reconstructed_user_data, stats


if __name__ == "__main__":
    main()
