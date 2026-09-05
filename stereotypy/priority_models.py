"""Optional specialized temporal heads for the cached priority experiment."""

def make_binary_model(dim=160):
    import torch
    class BinaryTemporal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project=torch.nn.Sequential(torch.nn.Linear(dim,64),torch.nn.GELU(),torch.nn.Dropout(.4))
            self.temporal=torch.nn.Sequential(torch.nn.Conv1d(64,64,3,padding=1),torch.nn.GELU(),torch.nn.Dropout(.4),
                                             torch.nn.Conv1d(64,64,3,padding=1),torch.nn.GELU())
            self.head=torch.nn.Linear(128,1)
        def forward(self,x):
            z=self.project(x);t=self.temporal(z.transpose(1,2))
            return self.head(torch.cat([z[:,3],t.mean(2)],1)).squeeze(-1)
    return BinaryTemporal()
