import logging
import os
import time
from datetime import datetime as dt

import numpy as np
import torch
from torch import optim

from data import Data
from model import ODEAutoEncoder, TrainerModel
from visualize import Visualizer


class Trainer:
    def __init__(self, model, optim, data, epochs, freq, folder=None, visualizer=None):
        assert isinstance(model, TrainerModel)
        self.model: TrainerModel = model
        self.optimizer = optim
        self.data = data
        self.epochs = epochs
        self.freq = freq

        if folder is None:
            self.folder = 'runs/model_' + dt.now().strftime("%m%d_%H%M")
        else:
            self.folder = folder

        if visualizer is None:
            self.visualizer = None
        else:
            self.visualizer = visualizer

        #logging.info(f'Instantiated trainer object with model {model}\n'
                    # f'and saving in folder {self.folder}\n'
                    # f'over {epochs} epochs logging every {freq} epoch')

    @classmethod
    def from_checkpoint(cls, model_class, path, epochs, freq, folder):
        obj = torch.load(path)

        model = model_class.from_checkpoint(path)
        optimizer = optim.Adam(model.get_params(), lr=0)
        optimizer.load_state_dict(obj['optimizer_state_dict'])

        data = Data.from_dict(obj['data'])

        visualizer = Visualizer(model, data, folder)

        trainer = cls(model, optimizer, data, epochs, freq,
                      folder=folder, visualizer=visualizer)

        #logging.info(f"Loaded model from {path}")

        # Get the version from next index where v is 'runs/model_1204_1635/ckpt/16_35_v3.pth'
        version = 0
        return trainer, version

#viene chiamato nel main con trainer.train
    def train(self, version=0):
        logging.info('Inizia il training')

        try:
            for epoch in range(self.epochs):  #iterazioni per epoche
                start = time.time()

                self.train_step(*self.data.get_train_data()) #calcola ELBO e aggiorna i pesi
                #get_train_data restituisce: samp_trajs_train, samp_ts
                #print("get_train_data restituisce: ", *self.data.get_train_data())

                self.validation_step(*self.data.get_val_data()) # calcola errore sulla validazione

                end = time.time()
                self.model.epoch_time.append(end - start)

                if epoch % self.freq == 0: #ogni tot epoche
                    if isinstance(self.model, ODEAutoEncoder):
                        logging.info(f'Current number of forward passes: {self.model.nfe_list[-1]}')

                    logging.info('Epoch: {}, train elbo: {:.4f}, validation elbo: {:.4f}, mean time per epoch: {:.4f}'
                                 .format(epoch,
                                         self.model.train_loss[-1],
                                         self.model.val_loss[-1],
                                         np.mean(self.model.epoch_time)))
                    print("Salviamo il modello")
                    self.save_model(version)
                    print("Chiamiamo lo step di visualizzazione")
                    self.visualize_step(version)
                    version += 1

#logging e visualizzazione dei risultati alla fine di tutto il ciclo for
            logging.info(f'Training finished after {epoch} epochs')
            self.save_model('final')
            print("Visualizzazione finale")
            self.visualize_final()


        except KeyboardInterrupt:
            logging.info('Stopped training due to interruption')
            self.save_model(version)
            self.visualize_final('interrupted')

    def visualize_step(self, version):
        if self.visualizer:
            self.visualizer.visualize_step(version)

    def visualize_final(self, version='final'):
        if self.visualizer:
            self.visualizer.visualize_final(version)

#viene chiamata in train con input samp_trajs_train, samp_ts oppure samp_trajs_val, samp_ts
    def train_step(self, x, t):
#INPUT:
#x: samp_trajs_train oppure samp_trajs_val
#t: samp_ts
        self.model.train() #attiva il modello in modalità training

        self.optimizer.zero_grad() #azzera i gradienti dell'ottimizzatore

        # Perform train step
        elbo = self.model.train_step(x, t)

        # Backwards pass and update optimizer and train loss
        elbo.backward() #calcola il gradiente della loss rispetto ai pesi del modello usando il backpropagation
        self.optimizer.step() #aggiorna i pesi del modello utilizzando l'ottimizzatore
        self.model.train_loss.append(-elbo.item()) #salva la loss nella lista train_loss del modello

    def validation_step(self, x, t):
        self.model.eval()

        with torch.no_grad():
            elbo = self.model.train_step(x, t)
            self.model.val_loss.append(-elbo.item())

    def save_model(self, version):
        folder = os.path.join(self.folder, 'ckpt')

        now = dt.now().strftime('%H_%M')
        ckpt_path = os.path.join(folder, f'{now}_v{version}.pth')
#
        save_dict = {
            'model_args': self.model.get_args(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'data': self.data.get_dict(),
            'train_loss': self.model.train_loss,
            'val_loss': self.model.val_loss
        }
        save_dict.update(self.model.get_state_dicts())
        torch.save(save_dict, ckpt_path)

        logging.info(f"Saved model at {ckpt_path}")
