from collections import deque
import string
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import TESNamesDataset
from models import TESLSTM
from generator import generate

# Configuration.
data_root = '/home/syafiq/Data/tes-names/'
charset = string.ascii_letters + '\'- '
max_length = 30
learning_rate = 0.0003
batch_size = 64
num_epochs = 100

# Prepare dataset/loader.
dataset = TESNamesDataset(data_root, charset, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# GPU execution.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Prepare model.
input_size = (
    len(dataset.race_codec.classes_) +
    len(dataset.gender_codec.classes_) +
    len(dataset.char_codec.classes_)
)
hidden_size = 128
output_size = len(dataset.char_codec.classes_)

model = TESLSTM(input_size, hidden_size, output_size)
model = model.to(device)

# Optimizer.
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = deque([], maxlen=100)

# Training.
for epoch in range(num_epochs):
    for batch_i, samples in enumerate(dataloader):
        model.zero_grad()

        t_race, t_gender, t_name = samples
        t_hidden, t_cell = model.init_hidden(t_race.size(0))

        t_race = t_race.to(device)
        t_gender = t_gender.to(device)
        t_name = t_name.to(device)
        t_hidden = t_hidden.to(device)
        t_cell = t_cell.to(device)

        loss = 0.

        for char_i in range(max_length - 1):
            t_char = t_name[:, char_i:char_i+1]
            t_output, t_hidden, t_cell = \
                model(t_race, t_gender, t_char, t_hidden, t_cell)

            targets = t_name[:, char_i+1:char_i+2].argmax(dim=2).squeeze()
            loss += criterion(t_output, targets)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_i % 100 == 0:
            print(generate('Argonian', 'Male', 'H', dataset, model, device))
            print('[%03d] %05d/%05d Loss: %.4f' % (
                epoch + 1,
                batch_i,
                len(dataset) // batch_size,
                sum(losses) / len(losses)
            ))

torch.save(model.state_dict(), 'model.pt')
