import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr = 0.1, momentum=0.9, weight_decay=0.0001)
        self.lr_schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40000, gamma=0.1)
        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        
        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = torch.tensor(y_train[shuffle_index], dtype=torch.long)

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            ### YOUR CODE HERE
            total_loss = 0
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                x_batch = curr_x_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
                y_batch = curr_y_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
                x_batch = self.process_data(x_batch)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()

                y_pred = self.network(x_batch.cuda())
                loss = self.loss(y_pred, y_batch.cuda())
                total_loss += loss
                loss.backward()
                self.optimizer.step()
                

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, total_loss/num_batches, duration))
            self.lr_schedule.step()
            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        x_temp = self.process_data(x)
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            
            count = 0
            for i in range(x.shape[0]):
                temp_x = x_temp[i]
                temp_x = temp_x.unsqueeze(0)
                pred = self.network(temp_x.cuda())
                pred = torch.argmax(pred, dim = 1)
                if y[i] == pred[0]:
                    count+=1

            print('Test accuracy: {:.4f}'.format(count/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))
        
    def process_data(self, batch):
        batch_size = len(batch)
        temp = np.zeros((batch_size, 3, 32, 32))
        for i in range(batch_size):
            temp[i] = parse_record(batch[i], True)
        temp = torch.tensor(temp, dtype=torch.float32)
        return temp