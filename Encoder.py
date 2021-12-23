import torch
from Trainer import BasicModel

def block(channels:     list,
          kernel_sizes: list,
          first:        bool = False,
          last:         bool = False):
    
    '''
    Function to build the convolution blocks of the encoder
    Returns a list with the layers of the convolution block

    Parameters:
    channels :    List whose items are lists that indicate the number of input channels and number of output channels 
                  of each convolutional layer of the block.
    kernel_sizes: List whose items indicate the kernel size of each convolutional layer of the block
    first:        If True, returns the first convolution block of the encoder. 
    last:         If True, returns the last convolution block of the encoder.
    '''
    
    if first:
        return torch.nn.Sequential(
                torch.nn.Conv1d(channels[0][0], channels[0][1], kernel_sizes[0], padding = 0, stride = 3),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU()
                )
    
    if last:
        return torch.nn.Sequential(
                torch.nn.Conv1d(channels[0][0], channels[0][1], kernel_sizes[0], padding = 1, stride = 1),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU()
                )
    
    sequentials = []
    
    for (c_in, c_out), k in zip(channels, kernel_sizes):
        sequentials.append(torch.nn.Sequential(
                            torch.nn.Conv1d(c_in, c_out, k, padding = 1, stride = 1),
                            torch.nn.BatchNorm1d(c_out),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool1d(k, stride = k)
                            )
        )
        
    return sequentials

class Encoder(BasicModel):
    """
    SampleCNN Encoder
    ==================================================================================================================
    1 One-dimensional convolution block with kernel size = 3 and ReLU activation.
    9 One-dimensional convolution blocks each with kernel size = 3, ReLU activation and max pooling with pool size = 3.
    1 One-dimensional convolution block with kernel size = 3 and ReLU activation.
    1 Linear layer that connects with the number of classes to predict.
    ==================================================================================================================
    """
    def __init__(self, n_classes=50):
        super().__init__()
        self.kernel_sizes = [3,3,3,3,3,3,3,3,3]
        self.channels = [[128, 128], [128, 128], [128, 256],
                        [256, 256], [256, 256], [256, 256], [256, 256], [256, 256],
                        [256, 512]]
        self.first_conv = block(channels = [[1,128]], kernel_sizes = [3], first = True)
        self.interm_conv = torch.nn.Sequential(*block(channels = self.channels, kernel_sizes = self.kernel_sizes))
        self.last_conv = block(channels=[[512,512]], kernel_sizes = [3], last = True)
        self.fc = torch.nn.Linear(512, n_classes)
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.interm_conv(x)
        x = self.last_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

#if __name__ == "__main__":
#    model = Encoder()
#    output = model(torch.randn(64, 1, 59049))
#    print(output.shape)