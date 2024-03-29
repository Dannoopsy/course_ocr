# +
import torch
import torch.nn as nn
import torch.nn.functional as F

class CellClassificationModel(nn.Module):
    def __init__(self, model, ncells, a=3, b=1):
        super(CellClassificationModel, self).__init__()
        self.model = model
        self.ncells = ncells
        self.a = a
        self.b = b
        self.loss_fn = KLDiscretLoss(use_target_weight=True, beta=10.,label_softmax=True)
    def forward(self, x):
        b = x.shape[0]
        return self.model(x).reshape(b, -1, ncells)
    def predict(self, logits):
#         logits = self.forward(x)
        activ = logits.argmax(-1)
        pred = (self.a * activ / self.ncells) - self.b
        return pred
    def loss(self, logits, y):
        y = (torch.clip((y + self.b) / self.a, 0, 1) * self.ncells).long()
        softlabels = xy2soft(y, self.ncells)
        b = logits.shape[0]
        predlabels = logits.reshape(-1, nump, 2, self.ncells)
        predlabels = (predlabels[:, :, 0], predlabels[:, :, 1])
        mask = torch.ones(b, nump).to(y.device)
        return self.loss_fn(predlabels, softlabels, mask)
        


class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax.
        label_softmax (bool): Whether to use Softmax on labels.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
    """

    def __init__(self, beta=1.0, label_softmax=False, use_target_weight=True):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.use_target_weight = use_target_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, mask, target_weight=None):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        num_joints = pred_simcc[0].size(1)
        loss = 0

        if self.use_target_weight and type(target_weight) != type(None):
            weight = target_weight.reshape(-1)
        else:
            weight = 1.

        for pred, target in zip(pred_simcc, gt_simcc):
            
            pred = pred * mask.view(-1, nump, 1) 
            target = target*mask.view(-1, nump, 1)
            
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))

            loss += self.criterion(pred, target).mul(weight).sum()

        return loss / num_joints

        

nump = 4
def xy2soft(coords, ncells):
    b = coords.shape[0]
    coords = coords.view(-1, nump, 2)
    l = torch.linspace(0, ncells - 1, ncells).view(1, 1, -1)
    ansx = l.expand(b, nump, -1).clone().cuda()
    ansy = l.expand(b, nump, -1).clone().cuda()
    ansx -= coords[:, :, 0].unsqueeze(2)
    ansy -= coords[:, :, 1].unsqueeze(2)
    d = ncells / 8
    ansx = - ansx.square() / d
    ansy = - ansy.square() / d
    ansx = torch.exp(ansx)
    ansy = torch.exp(ansy)
    return (ansx, ansy)

from torchvision.models import efficientnet_b5
# from torch import nn
ncells = 256
bb = efficientnet_b5(weights='DEFAULT')
bb.classifier = nn.Linear(in_features=2048, out_features=8 * 256, bias=True)
model = CellClassificationModel(bb, ncells)

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters())
l1 = nn.L1Loss(reduction='sum')
epochs = 20
scaler = torch.cuda.amp.GradScaler()
losses_train = []
losses_test = []


for i in range(epochs):
    loss_ep_train = 0
    loss_ep_test = 0
    model.train()
    
    for x, y in tqdm(trainloader):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x.cuda())
            loss = model.loss(logits, y.view(-1, 8).cuda())
        predicted = model.predict(logits)
        loss_l1 = l1(predicted, y.view(-1, 8).cuda())
        optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_ep_train += loss_l1
    
    model.eval()
    for x, y in tqdm(testloader):
        with torch.no_grad():
            logits = model(x.cuda())
            predicted = model.predict(logits)
            loss_l1 = l1(predicted, y.view(-1, 8).cuda())
            loss_ep_test += loss_l1
            
    losses_train.append(loss_ep_train.item() / len(trainset))
    losses_test.append(loss_ep_test.item() / len(testset))
    clear_output()
    plt.plot(losses_test, label='test')
    plt.plot(losses_train, label='train')
    plt.legend()
    plt.show()
