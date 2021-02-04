from model.unet import UNet
from model.loss import CrossEntropyWithLogits
from data.dataloader import create_dataset
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import time

num_epoches = 400
batch_size = 12
data_dir = "/userhome/Unet/unet/data/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataloader, val_dataloader = create_dataset(
    data_dir, repeat=1, train_batch_size = 12, augment=True)

model = UNet(1, 2).to(device)
criterion = CrossEntropyWithLogits().to(device)
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0005, eps=1e-08)

save_step = 400


##test data load time 
# print("get-100-epoch")
# load_s = time.time()
# for i in range(2):
#     for sample in train_dataloader:
#         print(sample["image"].shape)
#         print(sample["mask"].shape)
# load_e = time.time()
# print("load data time: ", load_e - load_s)


# TODO: Initialization the params
val_loss = -1
step_now = 0
total_loss = 0
scaler = GradScaler()
torch.cuda.synchronize()
start = time.time()
model.train()
## FP32 !
'''
for i in range(num_epoches):
    for id, sample in enumerate(train_dataloader):
        step_now += 1
        img = sample["image"]
        mask = sample["mask"]
        optimizer.zero_grad()
        pred = model(img.to(device))
        loss = criterion(pred, mask.to(device))
        loss.backward()
        optimizer.step()
        loss_cpu = loss.cpu().item()
        total_loss += loss_cpu
        print('step_now: {}, step loss: {}'.format(step_now,loss_cpu))
        if step_now % save_step == 0:
            val_loss = 0
            torch.save(model.state_dict(), "./step=%d" % (step_now))
'''
## FP16 !
for i in range(num_epoches):
    for id, sample in enumerate(train_dataloader):
        step_now += 1
        img = sample["image"]
        mask = sample["mask"]
        optimizer.zero_grad()
        with autocast():
            pred = model(img.to(device))
            loss = criterion(pred, mask.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_cpu = loss.cpu().item()
        print('step_now: {}, step loss: {}'.format(step_now,loss_cpu))
        total_loss += loss_cpu
        if step_now % save_step == 0:
            val_loss = 0
            torch.save(model.state_dict(), "./step=%d" % (step_now))

torch.cuda.synchronize()
end = time.time()
print("training time per step(ms): ", (end - start)*1000 / (num_epoches*2))
torch.save(model.state_dict(), "./result/step=all")
