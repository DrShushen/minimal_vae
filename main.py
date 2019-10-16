import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class DimTracer(object):
    def __init__(self, enabled, batch_log_interval, indentation_max):
        self.enabled = enabled
        self.current_batch_idx = 0
        self._batch_log_interval = batch_log_interval
        self._indentation_max = indentation_max

    def update_current_batch_index(self, batch_idx):
        if batch_idx:
            self.current_batch_idx = batch_idx

    def trace_dims(self, tensor, show_name):
        if self.enabled and \
           self.current_batch_idx % self._batch_log_interval == 0:
            print("{tensor_name}:{indentation}{tensor_size}".format(
                tensor_name=show_name,
                indentation="".join(
                    " " for _ in range(self._indentation_max - 
                        min(len(show_name), self._indentation_max))), 
                tensor_size=tensor.size()))


class VAE(nn.Module):
    
    def __init__(self, tracer=None):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.tracer = tracer

    def _trace_dims(self, tensor, show_name):
        if self.tracer:
            self.tracer.trace_dims(tensor, show_name)

    def encode(self, x):
        
        self._trace_dims(x, "x")

        fc1 = self.fc1(x)
        h1 = F.relu(fc1)
        self._trace_dims(fc1, "fc1")
        self._trace_dims(h1, "h1")
        
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        self._trace_dims(mu, "mu")
        self._trace_dims(logvar, "logvar")

        return mu, logvar

    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5*logvar)
        self._trace_dims(std, "std")

        eps = torch.randn_like(std)
        self._trace_dims(eps, "eps")

        z = mu + eps*std
        self._trace_dims(z, "z")

        return z

    def decode(self, z):

        fc3 = self.fc3(z)
        self._trace_dims(fc3, "fc3")

        h3 = F.relu(fc3)
        self._trace_dims(h3, "h3")

        fc4 = self.fc4(h3)
        self._trace_dims(fc4, "fc4")

        x_reconstr = torch.sigmoid(fc4)
        self._trace_dims(x_reconstr, "x_reconstr")

        return x_reconstr

    def forward(self, x):

        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_reconstr = self.decode(z)

        return x_reconstr, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, model, loader, optimizer, args, device, tracer):
    
    model.train()  # Sets the module in training mode.
    train_loss = 0
    
    if tracer:
        print("\nTrain...\n")

    for batch_idx, (data, _) in enumerate(loader):
        
        tracer.update_current_batch_index(batch_idx)

        data = data.to(device)
        tracer.trace_dims(data, "data")

        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)  # .forward()
        
        loss = loss_function(recon_batch, data, mu, logvar)
        tracer.trace_dims(loss, "loss")

        loss.backward()
        
        train_loss += loss.item()
        
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(loader.dataset)))


def test(epoch, model, loader, optimizer, args, device, tracer):
    
    model.eval()
    test_loss = 0
    
    if tracer:
        print("\nTest...\n")

    with torch.no_grad():
        for i, (data, _) in enumerate(loader):

            tracer.update_current_batch_index(i)

            data = data.to(device)
            tracer.trace_dims(data, "data")

            recon_batch, mu, logvar = model(data)  # .forward()
            
            loss = loss_function(recon_batch, data, mu, logvar)
            tracer.trace_dims(loss, "loss")

            test_loss += loss.item()
            
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [
                        data[:n], 
                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]
                    ])
                save_image(
                    comparison.cpu(),
                    "{}/reconstruction_{}.png".format(
                        args.results_subdir, 
                        epoch),
                    nrow=n)

    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":

    # Set up argument parsing:
    # Note: 
    # ... action='store_true', default=False ... - a way to get that argument 
    # to be False when not provided and True when provided. 
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='enables CUDA training')
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', 
        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status ' 
             '(default: 10)')
    parser.add_argument(
        '--trace-dims', action='store_true', default=False, 
        help='print dimensions of Tensors throughout the execution, '
             'for debug only (default: False)')
    parser.add_argument(
        '--results-subdir', type=str, default=None, metavar='DIR', 
        help='Output the result files into ./result/DIR/ (default: ./result/)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set up data loading:
    kwargs_loaders = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()),
        batch_size=args.batch_size, 
        shuffle=True, 
        **kwargs_loaders)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', 
            train=False, 
            transform=transforms.ToTensor()),
        batch_size=args.batch_size, 
        shuffle=True, 
        **kwargs_loaders)

    # Torch initialisation:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Initialise dimension tracer:
    tracer = DimTracer(
        enabled=args.trace_dims, 
        batch_log_interval=args.log_interval,
        indentation_max=15)

    # Initilise model and optimiser:
    model = VAE(tracer=tracer).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Ensure results output folder exists:
    args.results_subdir = "./results/{}".format(
        args.results_subdir if args.results_subdir is not None else "")
    if not os.path.exists(args.results_subdir):
        os.makedirs(args.results_subdir)

    # Train and test loop:
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, args, device, tracer)
        test(epoch, model, test_loader, optimizer, args, device, tracer)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                "{}/sample_{}.png".format(
                        args.results_subdir, 
                        epoch))
