import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import neetbox
from torchvision import transforms
from torch.utils.data import DataLoader
from neetbox import logger


class LinearVAE(nn.Module):
    def __init__(
        self,
        in_channel,
        latent_channel,
        reparam_channel,
        out_channel=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # [encode] input -> latent space
        self.input2latent = nn.Linear(in_channel, latent_channel)
        # [encode] latent space -> \mu
        self.latent2mu = nn.Linear(latent_channel, reparam_channel)
        # [encode] latent space -> \sigma
        self.latent2sigma = nn.Linear(latent_channel, reparam_channel)
        # [reparamererize] \mu and \sigma -> latent space
        self.reparam2latent = nn.Linear(reparam_channel, latent_channel)
        # [decode] latent space -> reconstruction
        self.latent2output = nn.Linear(latent_channel, out_channel or in_channel)

        self.relu_inplace = nn.ReLU(inplace=True)

    def encode(self, x):
        latent = self.input2latent(x)
        latent = self.relu_inplace(latent)
        mu = self.latent2mu(latent)
        sigma = self.latent2sigma(latent)
        return mu, sigma

    def reparamererize(self, mu, sigma):
        std = torch.exp(sigma / 2)
        epsilon = torch.randn_like(std)
        reparam = mu + epsilon * std
        return reparam

    def decode(self, reparam):
        latent = self.reparam2latent(reparam)
        latent = self.relu_inplace(latent)
        reconstructed = self.latent2output(latent)
        return F.sigmoid(reconstructed)

    def forward(self, x):
        mu, sigma = self.encode(x)
        reparam = self.reparamererize(mu, sigma)
        reconstructed = self.decode(reparam)
        return reconstructed, mu, sigma


config = {
    "epoch": 1000,
    "batch_size": 32,
    "input_width": 28,
    "input_height": 28,
    "lr": 3e-4,
}

# check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    dataset=datasets.MNIST(
        root="data/", train=True, transform=transforms.ToTensor(), download=True
    ),
    batch_size=config["batch_size"],
    shuffle=True,
)  # data loader for train

test_loader = DataLoader(
    dataset=datasets.MNIST(
        root="data/", train=True, transform=transforms.ToTensor(), download=True
    ),
    batch_size=1,
    shuffle=True,
)  # data loader for test


def loss_function(x, reconstructed_x, mu, sigma):
    BCE = F.binary_cross_entropy(reconstructed_x, x.view(-1, 784), reduction="sum")
    BCE /= x.size(0) * 784  # Normalize over batch size and image dimensions
    KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    KLD /= x.size(0) * 784  # Normalize over batch size and image dimensions
    return BCE, KLD


def train(model: LinearVAE, loss_func, train_loader: DataLoader):
    for _, (x, _) in enumerate(train_loader):
        input_dim = config["input_height"] * config["input_width"]
        x: torch.Tensor = x.to(device).view(-1, input_dim)  # flat it
        # model forwarding
        reconstructed, mu, sigma = model(x)
        # compute loss
        BCE, KLD = loss_func(x, reconstructed, mu, sigma)
        return BCE, KLD


def generate(
    model: LinearVAE, data_loader: DataLoader, target_digit: int, num_samples=7
):
    # stop train status
    model.eval()
    # pick input of target_digit
    input_image = None
    for x, label in data_loader:
        if target_digit == label:
            input_image = x
            break
    assert (
        input_image is not None
    ), f"something wrong, could not find target digit {target_digit} in data loader, the target digit should be in 1 to 9"
    # encode input image
    input_dim = config["input_height"] * config["input_width"]
    input_image = input_image.to(device)
    mu, sigma = model.encode(input_image.view(-1, input_dim))

    output_images = []
    for _ in range(num_samples):
        # add noise
        epsilon = torch.randn_like(sigma)
        reparam = mu + sigma * epsilon
        # decode from noise image
        reconstructed = model.decode(reparam)
        out = reconstructed.view(
            -1, 1, config["input_height"], config["input_width"]
        )  # reshape back to image
        output_images.append(out)

    # resmue model status
    model.train()
    return input_image, output_images

# for network architecture search
nas_configs = [
    {"latent": 32, "reparam": 4},
    {"latent": 64, "reparam": 8},
    {"latent": 128, "reparam": 16},
    {"latent": 256, "reparam": 32},
    {"latent": 512, "reparam": 64},
]

neetbox.add_hyperparams(
    {"train": config, "nas": nas_configs}
)  # just show hyperparams in web page that you can view

logger.log("starting Network Architecture Search")
for nas_cfg in neetbox.progress(nas_configs, name="Network Arch Search"):
    nas_name = f"latent({nas_cfg['latent']}), reparam({nas_cfg['reparam']})"
    input_dim = config["input_height"] * config["input_width"]
    model = LinearVAE(
        in_channel=input_dim,
        latent_channel=nas_cfg["latent"],
        reparam_channel=nas_cfg["reparam"],
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # train loop
    logger.info(f"start training for NAS config {nas_cfg}")
    for epoch in neetbox.progress(config["epoch"], name=f"train {nas_name}"):
        loss_reconstruct, kl_divergence = train(model, loss_function, train_loader)
        loss = loss_reconstruct + kl_divergence
        neetbox.add_scalar(
            f"loss reconstruct {nas_name}",
            epoch,
            loss_reconstruct.item(),
        )
        neetbox.add_scalar(f"kl_divergence {nas_name}", epoch, kl_divergence.item())
        neetbox.add_scalar(f"loss {nas_name}", epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test loop
    logger.info(f"start testing for NAS config {nas_cfg}")
    test_outputs = []
    for i in range(10):
        input_image, output_images = generate(model, test_loader, target_digit=i)
        showcase = torch.cat([input_image, torch.cat(output_images)])
        test_outputs.append(showcase)
    neetbox.add_image(
        name=f"test {nas_name}", image=torch.cat(test_outputs), dataformats="NCHW"
    )


@neetbox.action(name="generate digit")
def generate_digit(digit: int):
    input_image, output_images = generate(
        model, test_loader, target_digit=digit, num_samples=39
    )
    showcase = torch.cat([input_image, torch.cat(output_images)])
    neetbox.add_image(name=f"inference {nas_name}", image=showcase, dataformats="NCHW")


logger.info("serving the model via neetbox action. press ctrl+C to terminate.")
while True:
    time.sleep(1)