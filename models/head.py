import torch
import torch.nn as nn


class FusionHead(torch.nn.Module):
    def __init__(self, out_channels=256):
        super(FusionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 40, int(out_channels * 20)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_channels * 20), out_channels * 10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_channels*10, 1),
            )
        
    def forward(self, x_cell_embed, drug_embed):
        out = torch.cat((x_cell_embed, drug_embed), dim=1)
        out = self.fc(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)



