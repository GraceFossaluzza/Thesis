import matplotlib.pyplot as plt
import numpy as np
import torch
import os


from data import Data
from model import ODEAutoEncoder, LSTMAutoEncoder

plt.style.use('ggplot')

import logging
from datetime import datetime as dt

from sklearn.decomposition import PCA
import pandas as pd


class Visualizer:
    def __init__(self,model, data: Data, save_folder: str):
        self.model = model
        self.data = data
        self.save_folder = save_folder + 'png/'
        self.device = model.device

    def visualize_step(self, version):
        now = dt.now().strftime('%H_%M')
        fname = f'v{version}_{now}.png'

        self.plot_reconstruction(fname)

    def visualize_final(self, version, t_pos=np.pi, t_neg=np.pi):
        now = dt.now().strftime('%H_%M')
        fname = f'{version}_{now}'
        self.plot_reconstruction(fname + '_reconstruction.png', t_pos=t_pos, t_neg=t_neg)
        self.plot_loss_history(fname + '_loss_history.png')
        #tolgo momentaneamnete questi due plot
        #self.plot_reconstruction_grid(fname + '_reconstruction_grid.png', t_pos=t_pos, t_neg=t_neg)
        #self.plot_original_grid(fname + '_original_grid.png')

    def latent_vis(self, fname, z_traj_idx=None, test=False, label=None, saved_data=None):
        # We make sure that we plot for the test sample trajectories if test = True
        if test:
            samp_trajs, samp_ts, orig_trajs, _ = self.data.get_test_data()
            if label is None:
                label = self.data.get_test_labels()
        else:
            orig_trajs, samp_trajs, _, samp_ts = self.data.get_all_data()
            if label is None:
                label = self.data.get_all_labels()

        with torch.no_grad():
            if isinstance(self.model, ODEAutoEncoder):
                # We forward pass to extract pred_x and z0 (we will only use z0)
                if saved_data is not None:
                    qz0_mean, qz0_logvar, epsilon = self.model.encode(saved_data)
                else:
                    qz0_mean, qz0_logvar, epsilon = self.model.encode(samp_trajs)

                # Sample z0 (vector) from q(z0)
                z0 = self.model.sample_z0(epsilon, qz0_logvar, qz0_mean)
                print("z0 size", z0.size())

                pca = PCA(n_components=2)
                pca.fit(z0)
                print("Explained variance:", pca.explained_variance_ratio_)
                z0_red = pca.fit_transform(z0)

                # print(z0_red.shape)
                print(z0_red[:, 0].shape)
                print(len(label))

                # df=pd.DataFrame(z0_red,label)
                d = {'PC1': z0_red[:, 0], 'PC2': z0_red[:, 1], 'Label': label}
                df = pd.DataFrame(d)

                # print(len(df[df.Label==3].PC1))

                # z0 latent space plot
                if label is not None:
                    plt.figure()
                    # z0 samples in 2D
                    for i in np.unique(label):
                        colors = ["dummy", "#c4564f", "#51a66e", "#2697f0"]
                        plt.plot(df[df.Label == i].PC1, df[df.Label == i].PC2, 'o', color=colors[i],
                                 label=f' Spring type {i}', linewidth=2,
                                 zorder=1)
                    plt.legend()
                else:
                    plt.figure()
                    plt.plot(z0_red[:, 0], z0_red[:, 1], 'o', label='z0 samples in 2D', linewidth=2,
                             zorder=1)
                    plt.legend()

                #logging.info('Saved reconstruction at {}'.format(self.save_folder + fname))
                plt.savefig(self.save_folder + 'z0' + fname, dpi=250)
                if z_traj_idx is not None:
                    ts_rmse = torch.from_numpy(np.linspace(0., torch.max(samp_ts), num=len(samp_ts))).to(self.device)
                    pred_x, pred_z = self.model.decode(z0, ts_rmse, return_z=True)

                    # print(pred_z.size())
                    # print(pred_z)

                    pca_z = PCA(n_components=2)
                    pca_z.fit(pred_z[z_traj_idx, :, :])
                    # print(pca_z.explained_variance_ratio_)
                    pred_z_red = pca_z.fit_transform(pred_z[z_traj_idx, :, :])
                    # print(pred_z_red.shape)

                    stype = label[z_traj_idx]
                    print(stype)
                    plt.figure()
                    plt.plot(pred_z_red[:, 0], pred_z_red[:, 1], 'o', color=colors[stype],
                             label=f'latent traj Spring type {stype}', linewidth=2,
                             zorder=1)
                    plt.legend()

                    #logging.info('Saved reconstruction at {}'.format(self.save_folder + fname))
                    plt.savefig(self.save_folder + 'z_traj' + fname, dpi=250)

            elif isinstance(self.model, LSTMAutoEncoder):
                logging.info('Cannot sample latent space from Autoencoder baseline')


    def plot_reconstruction(self, fname, t_pos=np.pi, t_neg=np.pi, idx=0, test=True, toy=False):
        orig_trajs, samp_trajs, _, samp_ts = self.data.get_all_data()# We unwrap the trajectories from the data object
        #print("Valori asse x tempo: ",orig_trajs[idx, :, 0])

        # We make sure that we plot for the test sample trajectories if test = True
        if test: #la variabile test viene attivata per distinguere tra la modalità di addestramento o test
            samp_trajs, samp_ts, orig_trajs, _ = self.data.get_test_data()

        with torch.no_grad(): #PyTorch non memorizza i gradienti intermedi necessari per la backpropagation
            if isinstance(self.model, ODEAutoEncoder): #controlla se il modello è un ODEAutoEncoder

                qz0_mean, qz0_logvar, epsilon = self.model.encode(samp_trajs)# We forward pass to extract pred_x and z0 (we will only use z0)

                z0 = self.model.sample_z0(epsilon, qz0_logvar, qz0_mean)# Sample z0 (vector) from q(z0)

                # We generate new linspaces for extrapolation and negative extrapolation | We use the decode function to extract pred_x
                #ts_rmse = torch.from_numpy(np.linspace(1., torch.max(samp_ts).cpu().item(), num=len(samp_ts))).to(self.device)

                #ricostruzione
                ts_rec=samp_ts
                pred_x_rec=self.model.decode(z0,ts_rec)

            # Creiamo una finestra temporale per l'extrapolazione
                extrapolation_window = 10  # Numero di passi di extrapolazione
                num_extra_points = 3  # Quanti punti generare nell'extrapolazione

            # Creiamo la sequenza temporale per l'extrapolazione (pochi punti, non equidistanti)
                ts_max = torch.max(samp_ts).cpu().item()
                ts_extra = np.linspace(ts_max, ts_max + extrapolation_window, num=num_extra_points)
                ts_extra = torch.from_numpy(ts_extra).to(self.device)

            # Generiamo la traiettoria extrapolata con pochi punti
                pred_x_pos = self.model.decode(z0, ts_extra)

                #ts_pos = torch.from_numpy(np.linspace(0, torch.max(samp_ts).cpu().item() + t_pos, num=int(len(samp_ts)))).to(self.device)


                #pred_x_pos = self.model.decode(z0, ts_pos)
                #pred_x_rmse = self.model.decode(z0, ts_rmse)

                #val = int((t_pos / (torch.max(samp_ts) + t_pos)) * len(samp_ts)) #numero di punti di estrapolazione
                #pred_x_rec = pred_x_pos[:, :(len(samp_ts) - val), :]
                #pred_x_pos = pred_x_pos[:, (len(samp_ts) - val - 1):, :]
                #pred_x_rec=pred_x_pos[:,:]

#calcolo RMSE
                rmse_loss = self.RMSELoss(pred_x_rec, samp_trajs)
                logging.info(f'RMSE: {rmse_loss}')
#calcolo media
                mean_yhat, mean_y = self.compute_means(pred_x_rec, samp_trajs)
                print(f"Media di yhat: {mean_yhat.item()}, Media di y: {mean_y.item()}")
#calcolo della correlazione
                pearson_corr = self.compute_pearson_correlation(pred_x_rec, samp_trajs)
                print("La correlazione è pari a: ", pearson_corr)
#calcolo della varianza
                var_yhat, var_y=self.compute_var(pred_x_rec,samp_trajs)
                print("La varianza di yhat: ",var_yhat, "La varianza di y: ",var_y)
#calcolo della potenza
               # power=self.compute_power_bands()

                #salviamo le metriche calcolate
                # Salva i risultati in un file numpy alla fine del training
                metrics = {
                    'rmse_loss': rmse_loss.item(),
                    'mean_yhat': mean_yhat.item(),
                    'mean_y': mean_y.item(),
                    'pearson_corr': pearson_corr.item(),
                    'var_yhat': var_yhat.item(),
                    'var_y': var_y.item()
                }

                #Metrics
                metrics_path = f"/content/drive/MyDrive/Tesi/Reconstructions/windowsize{window_size}/SUB{subject}/condition{condition}/Metrics/band{band}/channel{channel}/final_metrics.npz"

# Estrai solo la directory (senza il file finale)
                metrics_dir = os.path.dirname(metrics_path)

# Crea tutte le directory necessarie
                os.makedirs(metrics_dir, exist_ok=True)
                np.savez(metrics_path, **metrics)
                print(f"Metriche salvate in: {metrics_path}")


                pred_x_pos = pred_x_pos.cpu().detach().numpy()
                pred_x_rec = pred_x_rec.cpu().detach().numpy()

                # We plot only the first trajectory
                orig_trajs = orig_trajs.cpu().detach()
                samp_trajs = samp_trajs.cpu().detach()

                if (t_neg > 0):
                    ts_neg = torch.from_numpy(np.linspace(-t_neg, 0., num=int(len(samp_ts) / 8))[::-1].copy()).to(
                        self.device)
                    pred_x_neg = torch.flip(self.model.decode(z0, ts_neg), dims=[1]).cpu().detach().numpy()

                    plt.figure()
                    plt.plot(orig_trajs[idx, :, 0].cpu().numpy(), orig_trajs[idx, :, 1].cpu().numpy(), 'g', label='True trajectory', linewidth=2,
                             zorder=1)

                    if isinstance(pred_x_rec, torch.Tensor):
                        pred_x_rec = pred_x_rec.cpu().numpy()

                    plt.plot(pred_x_rec[idx, :, 0], pred_x_rec[idx, :, 1], '-o', color='r', markersize=3,label='Reconstruction', zorder=3)

                    if isinstance(pred_x_pos, torch.Tensor):
                        pred_x_pos = pred_x_pos.cpu().numpy()

                    #plt.plot(pred_x_pos[idx, :, 0], pred_x_pos[idx, :, 1], '-o', color='c', markersize=3,
                    #label='Learned trajectory (t>0)', zorder=2)

                  # Se pred_x_neg è un tensore, converti in NumPy
                    if isinstance(pred_x_neg, torch.Tensor):
                        pred_x_neg = pred_x_neg.cpu().numpy()

                    #plt.plot(pred_x_neg[idx, :, 0], pred_x_neg[idx, :, 1], '-o', color='c', markersize=3,
                    #label='Learned trajectory (t<0)', zorder=2)
                    plt.scatter(samp_trajs[idx, :, 0].cpu().numpy(), samp_trajs[idx, :, 1].cpu().numpy(), color='b', label='Sampled data', s=10,
                                zorder=2)
                    plt.legend()
                else:
                    plt.figure()
                    plt.plot(orig_trajs[idx, :, 0].cpu().numpy(), orig_trajs[idx, :, 1].cpu().numpy(), 'g', label='True trajectory', zorder=1)
                    plt.plot(pred_x_rec[idx, :, 0], pred_x_rec[idx, :, 1], '-o', color = 'r', markersize = 3, label='Reconstruction', zorder=3)
                    #plt.plot(pred_x_pos[idx, :, 0].cpu().numpy(), pred_x_pos[idx, :, 1].cpu().numpy(), '-o', color='c', markersize=3,
                             #label='Learned trajectory (t>0)', zorder=3)
                    plt.scatter(samp_trajs[idx, :, 0].cpu().numpy(), samp_trajs[idx, :, 1].cpu().numpy(), color='b', label='Sampled data', s=3,
                                zorder=2)
                    plt.legend()

            elif isinstance(self.model, LSTMAutoEncoder):
                pred_x = self.model.forward(samp_trajs)

                plt.figure()
                plt.plot(orig_trajs[idx, orig_trajs[idx, :, 0] >= 0, 0].cpu().numpy(), orig_trajs[idx, orig_trajs[idx, :, 0] >= 0, 1].cpu().numpy(), 'g', label='true trajectory', zorder=1)
                plt.plot(pred_x[idx, :, 0].cpu().numpy(), pred_x[idx, :, 1].cpu().numpy(), 'r', label='learned trajectory (t>0)', zorder=3)
                plt.scatter(samp_trajs[idx, :, 0].cpu().numpy(), samp_trajs[idx, :, 1].cpu().numpy(), color='b', label='sampled data', s=3,
                            zorder=2)
                plt.legend()

        if not os.path.exists(self.save_folder):
          os.makedirs(self.save_folder)
        #logging.info('Saved reconstruction at {}'.format(self.save_folder + fname))
        plt.savefig(self.save_folder + fname, dpi=250)

        # Definisci la cartella per salvare le finestre ricostruite
        save_dir = f"./Reconstructions/windowsize{window_size}/SUB{subject}/condition{condition}/Metrics/band{band}/channel{channel}/final_metrics.npz"
        os.makedirs(save_dir, exist_ok=True)

        if crossed==False: #per salvare solo la ricostruzione finale
          np.save(f"{save_dir}/final_reconstruction.npy", pred_x_rec if isinstance(pred_x_rec, np.ndarray) else pred_x_rec.cpu().detach().numpy())

        
        if crossed == True:  # per salvare tutte le ricostruzioni
            npy_name = fname.replace("reconstruction", "final_reconstruction").replace(".png", ".npy")
            reconstruction = pred_x_rec[idx]  # shape: (window_size, 2)
    
    # Path locale relativo
            save_dir = os.path.join(
            "./Reconstructions",
            f"windowsize{window_size}",
            f"SUB{subject}",
            f"condition{condition}",
            f"band{band}",
            f"channel{channel}",
            f"idx{idx}"
            )
    
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, npy_name), reconstruction)

        #np.save(f"./final_reconstruction_{band}_{channel}.npy",pred_x_rec if isinstance(pred_x_rec, np.ndarray) else pred_x_rec.cpu().detach().numpy())



    def plot_reconstruction_grid(self, fname, t_pos=np.pi, t_neg=np.pi, size=5, test=False):
        # We unwrap the trajectories from the data object
        orig_trajs, samp_trajs, _, samp_ts = self.data.get_all_data()

        # We make sure that we plot for the test sample trajectories if test = True
        if test:
            samp_trajs, samp_ts, orig_trajs, _ = self.data.get_test_data()

        with torch.no_grad():
            if isinstance(self.model, ODEAutoEncoder):
                # We forward pass to extract pred_x and z0 (we will only use z0)
                qz0_mean, qz0_logvar, epsilon = self.model.encode(samp_trajs)

                # Sample z0 (vector) from q(z0)
                z0 = self.model.sample_z0(epsilon, qz0_logvar, qz0_mean)

                # We generate new linspaces for extrapolation and negative extrapolation | We use the decode function to extract pred_x
            # Converte torch.max(samp_ts) in un numero scalare prima di passarlo a NumPy
                max_samp_ts = torch.max(samp_ts).cpu().item()

                ts_pos = torch.from_numpy(np.linspace(0., max_samp_ts + t_pos, num=len(samp_ts))).to(self.device)
                pred_x_pos = self.model.decode(z0, ts_pos).cpu().detach().numpy()

                if t_neg > 0:
                    ts_neg = torch.from_numpy(np.linspace(-t_neg, 0., num=int(len(samp_ts) / 8))[::-1].copy()).to(
                        self.device)
                    pred_x_neg = torch.flip(self.model.decode(z0, ts_neg), dims=[1]).cpu().detach().numpy()

                # Define extrapolation
                #val = int((t_pos / (torch.max(samp_ts) + t_pos)) * len(samp_ts))
                val=0
                pred_x_rec = pred_x_pos[:, :(len(samp_ts) - val), :]
                pred_x_pos = pred_x_pos[:, (len(samp_ts) - val - 1):, :]

                orig_trajs = orig_trajs.cpu().detach()
                samp_trajs = samp_trajs.cpu().detach()

                plt.figure(figsize=(15, 15))
                for i in range(size ** 2):
                    # We scale all y values to be 0:1
                    min_traj_y = np.min(orig_trajs.numpy()[i, :, 1])
                    max_traj_y = np.max(orig_trajs.numpy()[i, :, 1])
                    pred_x_rec_plt_y = (pred_x_rec[i, :, 1] - min_traj_y) / (max_traj_y - min_traj_y)
                    pred_x_pos_plt_y = (pred_x_pos[i, :, 1] - min_traj_y) / (max_traj_y - min_traj_y)
                    orig_trajs_plt_y = (orig_trajs[i, :, 1] - min_traj_y) / (max_traj_y - min_traj_y)

                    # We scale all x values to be 0:1
                    min_traj_x = np.min(orig_trajs.numpy()[i, :, 0])
                    max_traj_x = np.max(orig_trajs.numpy()[i, :, 0])
                    pred_x_rec_plt_x = (pred_x_rec[i, :, 0] - min_traj_x) / (max_traj_x - min_traj_x)
                    pred_x_pos_plt_x = (pred_x_pos[i, :, 0] - min_traj_x) / (max_traj_x - min_traj_x)
                    orig_trajs_plt_x = (orig_trajs[i, :, 0] - min_traj_x) / (max_traj_x - min_traj_x)

                    plt.subplot(size, size, i + 1)
                    plt.plot(pred_x_rec_plt_x, pred_x_rec_plt_y, '-o', color='r', markersize=1, label='Reconstruction',
                             zorder=3)
                    plt.plot(orig_trajs_plt_x, orig_trajs_plt_y, color='g', linewidth=1, label='True trajectory',
                             markersize=1, zorder=1)
                    #if t_pos > 0:
                        #plt.plot(pred_x_pos_plt_x, pred_x_pos_plt_y, '-o', color='c', markersize=3,
                                 #label='Learned trajectory (t>0)', zorder=2)
                    #if t_neg > 0:
                        #pred_x_neg_plt_y = (pred_x_neg[i, :, 1] - min_traj_y) / (max_traj_y - min_traj_y)
                        #pred_x_neg_plt_x = (pred_x_neg[i, :, 0] - min_traj_x) / (max_traj_x - min_traj_x)
                        #plt.plot(pred_x_neg_plt_x, pred_x_neg_plt_y, '-o', color='c', markersize=3,
                                 #label='Learned trajectory (t<0)', zorder=2)

                plt.legend(
                    #['Reconstruction', 'True trajectory', 'Learned trajectory (t>0)', 'Learned trajectory (t<0)'])
                    ['Reconstruction', 'True trajectory'])

            elif isinstance(self.model, LSTMAutoEncoder):
                pred_x = self.model.forward(samp_trajs)

                plt.figure(figsize=(15, 15))

                for i in range(size ** 2):
                    plt.subplot(size, size, i + 1)
                    plt.plot(orig_trajs[i, :, 0], orig_trajs[i, :, 1], 'g', label='true trajectory', zorder=1)
                    plt.plot(pred_x[i, :, 0], pred_x[i, :, 1], 'r', label='learned trajectory (t>0)', zorder=3)
                    plt.scatter(samp_trajs[i, :, 0], samp_trajs[i, :, 1], color='b', label='sampled data', s=3,
                                zorder=2)
                plt.legend()

            #logging.info('Saved reconstruction grid plot at {}'.format(self.save_folder + fname))
            plt.savefig(self.save_folder + fname, dpi=250)

    def plot_original_grid(self, fname):
        # We unwrap the trajectories from the data object
        orig_trajs, _, _, _ = self.data.get_all_data()

        orig_trajs = orig_trajs.cpu().detach()

        plt.figure(figsize=(15, 15))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.scatter(x=orig_trajs[i, :, 0], y=orig_trajs[i, :, 1], color='b')

        plt.legend(['orig_trajs'])

        #logging.info('Saved original grid plot at {}'.format(self.save_folder + fname))
        plt.savefig(self.save_folder + fname, dpi=250)

    def plot_loss_history(self, fname):
        plt.figure(figsize=(15, 15))
        plt.plot(self.model.train_loss, color='b')
        plt.plot(self.model.val_loss, color='r')

        plt.legend(['train ELBO', 'validation ELBO'])

        #logging.info('Saved loss plot at {}'.format(self.save_folder + fname))
        plt.savefig(self.save_folder + fname, dpi=250)

    def plot_latent_space(self, fname):
        # We unwrap the trajectories from the data object
        orig_trajs, samp_trajs, orig_ts, samp_ts = self.data.get_all_data()

        # We forward pass to extract pred_x and z0
        pred_x, z0 = self.model.forward(samp_trajs, samp_ts, return_z0=True)

        # We use the decode function to extract pred_z
        pred_z = self.model.decode(z0, samp_ts, return_z=True)

        # Dunno if this is necessary
        pred_z = pred_z.cpu().detach().numpy()

        # We create the plot
        plt.figure(figsize=(15, 15))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.scatter(x=pred_z[i, :, 0], y=pred_z[i, :, 1], color='r')
        plt.legend(['pred_z,'])
        plt.savefig(fname, dpi=250)
        plt.show()

    def plot_extrapolation(self):
        pass

    def RMSELoss(self, yhat, y):
        assert type(yhat) == torch.Tensor
        assert type(y) == torch.Tensor
        if y.device != yhat.device:
          y = y.to(yhat.device)

        return torch.sqrt(torch.mean((yhat - y) ** 2))

    def compute_means(self, yhat, y):
      #verifica che entrambi gli input siano tensori PyTorch
        assert isinstance(yhat, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        mean_yhat = torch.mean(yhat)
        mean_y = torch.mean(y)

        return mean_yhat, mean_y

    def compute_var(self,yhat,y):
        assert isinstance(yhat,torch.Tensor)
        assert isinstance(y,torch.Tensor)

        var_yhat = torch.var(yhat, unbiased=False)  # Varianza di yhat
        var_y = torch.var(y, unbiased=False)  # Varianza di y

        return var_yhat, var_y

    def compute_pearson_correlation(self, yhat, y): #misura la correlazione lineare tra due tensori (valore tra -1 e 1)
    #controlla che entrambi gli input siano tensori PyTorch
        assert isinstance(yhat, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        if y.device != yhat.device:
          y = y.to(yhat.device)

        #calcolo medie
        x_mean = torch.mean(yhat)
        y_mean = torch.mean(y)
        #print("La forma di yhat e y è: ", yhat.shape, y.shape)

        #centratura
        x_diff = yhat - x_mean
        y_diff = y - y_mean

        #calcolo del numeratore della formula di Pearson
        numerator = torch.sum(x_diff * y_diff)
        denominator = torch.sqrt(torch.sum(x_diff ** 2) * torch.sum(y_diff ** 2)) #radice quadrata del prodotto delle varianze

        correlation = numerator / (denominator + 1e-8)  # Per evitare divisioni per zero
        return correlation


    def computeRMSE_VAE(self, samp_trajs, samp_ts):
        with torch.no_grad():
            pred_x_rmse = self.model.forward(samp_trajs, samp_ts)
            rmse_loss = self.RMSELoss(pred_x_rmse, samp_trajs)

            return (rmse_loss.cpu().detach().numpy(), pred_x_rmse)

    def computeRMSE_AE(self, samp_trajs):
        with torch.no_grad():
            pred_x_rmse = self.model.forward(samp_trajs)
            rmse_loss = self.RMSELoss(pred_x_rmse, samp_trajs)

            return (rmse_loss.cpu().detach().numpy(), pred_x_rmse)
