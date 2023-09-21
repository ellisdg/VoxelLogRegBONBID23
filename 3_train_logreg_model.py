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

zmap = torch.from_numpy(zmap_array)[:, None].cuda()

label = torch.from_numpy(label_array)[:, None].to(torch.double)
label.requires_grad = True
label = label.cuda()

class Model(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(shape), requires_grad=True)
    
    def forward(self, x):
        return torch.sigmoid(-x - self.weights)

model = Model((1, *zmap.shape[1:])).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# loss = DiceLoss(include_background=True).cuda()
loss = torch.nn.BCELoss().cuda()

print(zmap.dtype)
print(label.dtype)

for i in range(100):
    pred = model(zmap)
    print(pred.dtype)
    i_loss = loss(pred, label)
    print(i_loss)
    i_loss.backward()
    optimizer.step()
