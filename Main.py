import argparse
import Data_augmentation as DA
import Encoder
import Magnatagatune
import Trainer
import torch

from simclr import SimCLR
from torch.utils.data import DataLoader




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    args = parser.parse_args()
    
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformations = [(DA.RandomResize, 1),
                       (DA.PolInversion, 0.8),
                       (DA.AddGausNoise, 0.01),
                       (DA.GainReduction, 0.3),
                       (DA.FrequencyFilter, 0.8),
                       (DA.Delay, 0.3),
                       (DA.P_Shift, 0.6),
                       (DA.Rvrb, 0.6)
                      ]

    train_dataset = Magnatagatune.Dataset(args.root, subset='train', trans=transformations)
    valid_dataset = Magnatagatune.Dataset(args.root, subset='val', trans=transformations)
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    enc = Encoder.Encoder()
    model = SimCLR(enc, 64, 512)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    """
    Trainer.fit(train_dataloader = train_loader,
              test_dataloader = valid_loader,
              optimizer = optimizer,
              epochs = 10)
              
    """
    