from pathlib import Path
import pandas as pd
from tools.preprocessing import MFCC

import numpy as np
import scipy
# import matplotlib.pyplot as plt
import sklearn.mixture
import sklearn.model_selection
from scipy.io import wavfile as wav
from os import listdir
from collections import defaultdict
from sklearn.metrics import log_loss
from ruamel.yaml import YAML

import pickle
from tqdm import tqdm

class GMM():

  def __init__(self, train_set, test_set, config_file, audio_preprocessor):
    self.train = train_set
    self.test = test_set
    self.config_file = config_file

    self.mfcc_preprocessor = MFCC(config_file, audio_preprocessor)

    if "GMM" in config_file.keys():
      config_file = self.config_file["GMM"]
      print("******\nLoaded GMM config file.\n******")
      self.if_train = config_file["if_train"] if "if_train" in config_file.keys() else True
      self.n_components = config_file["n_components"] if "n_components" in config_file.keys() else 20
      self.covariance_type = config_file["covariance_type"] if "covariance_type" in config_file.keys() else 'full' 
      self.tol = config_file["tol"] if "tol" in config_file.keys() else 0.001
      self.reg_covar = config_file["reg_covar"] if "reg_covar" in config_file.keys() else 1e-06
      self.max_iter = config_file["max_iter"] if "max_iter" in config_file.keys() else 100
      self.n_init = config_file["n_init"] if "n_init" in config_file.keys() else 5
      self.init_params = config_file["init_params"] if "init_params" in config_file.keys() else 'kmeans'
      self.weights_init = config_file["weights_init"] if "weights_init" in config_file.keys() else None
      self.means_init = config_file["means_init"] if "means_init" in config_file.keys() else None
      self.precisions_init = config_file["precisions_init"] if "precisions_init" in config_file.keys() else None
      self.random_state = config_file["random_state"] if "random_state" in config_file.keys() else 10
      self.warm_start = config_file["warm_start"] if "warm_start" in config_file.keys() else False
      self.verbose = config_file["verbose"] if "verbose" in config_file.keys() else 0
      self.verbose_interval = config_file["verbose_interval"] if "verbose_interval" in config_file.keys() else 10
        
    else:                
      print("Incorrect config file")
      raise 


  def load_models(self, path_to_model):
    with open(path_to_model, 'rb') as handle:
      self.gmm_train = pickle.load(handle)


  def prepare_models(self):
    if not self.if_train:
      self.train = self.load_models((Path(__file__).parent / "models") / "gmm_models.pickle")
  
    else:
      self.train = self.train_gmm()


  def GaussianMix(self):
    gmm=sklearn.mixture.GaussianMixture(n_components=self.n_components, 
                                        covariance_type=self.covariance_type, 
                                        tol=self.tol, 
                                        reg_covar=self.reg_covar,
                                        max_iter=self.max_iter, 
                                        n_init=self.n_init, 
                                        init_params=self.init_params, 
                                        weights_init=self.weights_init,
                                        means_init=self.means_init, 
                                        precisions_init=self.precisions_init, 
                                        random_state=self.random_state,
                                        warm_start=self.warm_start, 
                                        verbose=self.verbose, 
                                        verbose_interval=self.verbose_interval)
    return gmm   


  def gmm_commands(self, set_to_process, if_save):

    gmm = self.GaussianMix()

    commands = pd.unique(set_to_process.command)

    gmm_dict = dict()

    print("***\nTRAIN\n***")
    for i, command_ in enumerate(tqdm(commands)):

      command_set = list(set_to_process[set_to_process.command == command_].path)

      mfcc_command = np.array([])
      for wavp in command_set:
        mfcc_tmp = self.mfcc_preprocessor.compute_mfcc(signal_path = wavp)

        if not np.any(np.isnan(mfcc_tmp)):
          if len(mfcc_command) > 0:
            mfcc_command = np.concatenate([mfcc_command, mfcc_tmp], axis = 1)
          else:
            mfcc_command = mfcc_tmp

      gmm_dict[command_] = gmm.fit(mfcc_command.T)

    if if_save:
      if not (Path(__file__).parent / "models").exists():
        Path.mkdir(Path(__file__).parent / "models")
      
      path_to_save = (Path(__file__).parent / "models") / "gmm_models.pickle"
      
      with open(path_to_save, 'wb') as handle:
        pickle.dump(gmm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return gmm_dict


  def train_gmm(self, if_save = True):

    ### train
    self.gmm_train = self.gmm_commands(self.train, if_save)

    return self.gmm_train


  def get_prediction(self, path_to_wave):
  
    mfcc = self.mfcc_preprocessor.compute_mfcc(signal_path = path_to_wave)
    mfcc = np.nan_to_num(mfcc)
    p_score = []
    prediction = "unknown"
    models = self.gmm_train

    for gmm_key in models.keys():
      p_score.append(models[gmm_key].score(mfcc.T))

    prediction = list(self.gmm_train.keys())[p_score.index(max(p_score))]
    return prediction, max(p_score)

 
  def get_prediction_set(self, path_to_save):
    results = dict()
    results["prediction"] = []
    results["prediction_score"] = []
    results["ground_truth"] = []

    print("***\nTEST\n***")
    for index, row in tqdm(self.test.iterrows(), total=self.test.shape[0]):
      
      prediction, p_score = self.get_prediction(row["path"])
      results["prediction"].append(prediction)
      results["prediction_score"].append(p_score)
      results["ground_truth"].append(row["command"])
      print(results)
      break
 
    pd.DataFrame(results).to_csv(path_to_save, index = False)

    return pd.DataFrame(results)

  


    



