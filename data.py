import logging
import numpy as np
import torch


class Data:
    def __init__(self, orig_trajs, samp_trajs, orig_ts, samp_ts, labels=None, device=None):
        if device is None: #se il device non è stato assegnato allora usa la CPU
            device = 'cpu'

        try:
            orig_trajs = torch.from_numpy(orig_trajs).float()
            samp_trajs = torch.from_numpy(samp_trajs).float()
            samp_ts = torch.from_numpy(samp_ts).float()
            orig_ts = torch.from_numpy(orig_ts).float()

        except Exception:
            logging.warning('Inputs cannot be converted to torch (already a torch obj?)\n'
                            f'Types: {type(orig_trajs), type(samp_trajs), type(orig_ts), type(samp_ts)} ')

        #sposta sul device i parametri
        self.samp_ts = samp_ts.to(device)
        self.samp_trajs = samp_trajs.to(device)
        self.orig_ts = orig_ts.to(device)
        self.orig_trajs = orig_trajs.to(device)
        self.labels = labels
        #splitta i dati tra la parte di training e validation
        self.split(orig_trajs, samp_trajs, labels)

#aggiunta ma da capire se serve o no
    def get_samp_ts(self):
        return self.samp_ts

    @classmethod #possiamo chiamare questo metodo direttamente dalla classe, senza crearne un'istanza
    #NB: nei metodi di classe si usa cls per riferirsi alla classe stessa in modo generico
    def from_func(cls, func, device, **args):
        return cls(*func(**args), device=device)
        #Questo metodo chiama la funzione func (in questo caso load_data).load_data restituisce quattro tensori. Questi tensori vengono passati come argomenti alla classe Data, che li usa per creare un'istanza dell'oggetto Data.

    #divide il dataset in 3 insiemi: 60-20-20
    def split(self, orig_trajs, samp_trajs, labels, train_split=0.6, val_split=0.2):
        # We split the data across the spring dimension [nr. springs, nr. samples, values]
        train_int = int(train_split * orig_trajs.shape[0])  # X% of the data length for training, posizione dell'elemento dove termina il training set
        val_int = int((train_split + val_split) * orig_trajs.shape[0])  # X% more for validation, posizione dell'elemento dove termine il validation set

        #divisione delle traiettorie originali
        self.orig_trajs_train, self.orig_trajs_val, self.orig_trajs_test = (orig_trajs[:train_int, :, :],
                                                                            orig_trajs[train_int:val_int, :, :],
                                                                            orig_trajs[val_int:, :, :])
        #divisione delle traiettorie campionate
        self.samp_trajs_train, self.samp_trajs_val, self.samp_trajs_test = (samp_trajs[:train_int, :, :],
                                                                            samp_trajs[train_int:val_int, :, :],
                                                                            samp_trajs[val_int:, :, :])
        #divisione delle etichette, se presenti
        if labels:
            self.labels_train, self.labels_val, self.labels_test = (labels[:train_int],
                                                                    labels[train_int:val_int],
                                                                    labels[val_int:])

    def get_train_data(self):
        return self.samp_trajs_train, self.samp_ts

    def get_val_data(self):
        return self.samp_trajs_val, self.samp_ts

    def get_test_data(self):
        return self.samp_trajs_test, self.samp_ts, self.orig_trajs_test, self.orig_ts

    def get_all_data(self):
        return self.orig_trajs, self.samp_trajs, self.orig_ts, self.samp_ts

    def get_train_labels(self):
        return self.labels_train

    def get_val_labels(self):
        return self.labels_val

    def get_test_labels(self):
        return self.labels_test

    def get_all_labels(self):
        return self.labels

    @classmethod #metodo che non necessita di un'istanza della classe per essere chiamato
    def from_dict(cls, dict, device=None): #a partire da un dizionario dict crea un'istanza della classe
        labels = None
        if 'labels' in dict:
            labels = dict['labels']
            logging.info("Loading data labels")

        return cls(
            dict['orig_trajs'],
            dict['samp_trajs'],
            dict['orig_ts'],
            dict['samp_ts'],
            labels,
            device
        )

    def get_dict(self):
        return {
            'orig_trajs': self.orig_trajs,
            'samp_trajs': self.samp_trajs,
            'orig_ts': self.orig_ts,
            'samp_ts': self.samp_ts,
            'labels': self.labels,
        }
