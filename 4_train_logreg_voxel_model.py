import numpy, torch, nibabel as nib, glob, os
from monai.losses import DiceLoss
import numpy as np

zmap_array = list()
label_array = list()
for zmap_fn in glob.glob("./antsoutputs/final/*/zmap.nii.gz"):
    zmap = nib.load(zmap_fn)
    zmap_array.append(zmap.get_fdata())
    label_fn = zmap_fn.replace("zmap", "label")
    label = nib.load(label_fn)
    label_array.append(label.get_fdata() > 0.5)


zmap_array = np.stack(zmap_array)
label_array = np.stack(label_array)

zmap = torch.from_numpy(zmap_array)[:, None].to(torch.float).cuda()

label = torch.from_numpy(label_array)[:, None].to(torch.float)
label.requires_grad = True
label = label.cuda()

class Model(torch.nn.Module):
    def __init__(self, n_voxels):
        super().__init__()
        self.conv = torch.nn.Conv1d(n_voxels, n_voxels, 1, 1, groups=n_voxels)
    
    def forward(self, x):
        shape = x.shape
        return torch.sigmoid(self.conv(x.flatten(start_dim=2).squeeze()[..., None])).reshape(shape)

model = Model(zmap.shape[2] * zmap.shape[3] * zmap.shape[4]).cuda()

if os.path.exists("logistic_regression_voxel.pth"):
    logreg_weights = torch.load("./logistic_regression_voxel.pth")
    model.conv.weight.data[:] = logreg_weights['conv.weight']
    model.conv.bias.data[:] = logreg_weights['conv.bias']
else:    
    logreg_weights = torch.load("./logistic_regression.pth")
    model.conv.weight.data[:] = logreg_weights['linear.weight'].squeeze()
    model.conv.bias.data[:] = logreg_weights['linear.bias'].squeeze()
# print(model.state_dict())
# print(model.conv.weight.requires_grad)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

dice_loss = DiceLoss(include_background=True).cuda()
bce_loss = torch.nn.BCELoss().cuda()

# print(zmap.dtype)
# print(label.dtype)
best = torch.inf

for i in range(10000):
    optimizer.zero_grad()
    pred = model(zmap)
    _dloss = dice_loss(pred, label)
    _bloss = bce_loss(pred, label)
    i_loss = _dloss + _bloss
    print(_dloss.data, _bloss.data, i_loss.data)
    if _dloss < best:
        best = _dloss
        torch.save(model.state_dict(), "logistic_regression_voxel.pth")
    i_loss.backward()
    optimizer.step()
    
