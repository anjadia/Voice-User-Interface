from pathlib import Path
from asr.tools.preprocessing import compute_mfcc

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn.mixture
import sklearn.model_selection
from scipy.io import wavfile as wav
from os import listdir
from collections import defaultdict
from sklearn.metrics import log_loss

"""
Jest tu trochę bałagan i nie będę pisać po angielsku, bo jest za późno.
Nie jestem w stanie pojąć jak miałabym zrobić listę GMMów różnych,
skoro wrzuca się do nich jakieś stałe liczby, a dopiero przy fittowaniu
dostaje macierze MFCC.

"""
# artefakt, ale może do czegoś się przyda.
lista_fraz = ["wlacz", "wylacz", "zgas", "zaswiec", "zamknij","otworz","start","stop","zaslon","odslon","odbierz","odrzuc","naprzod","wstecz",
              "styczen","luty","marzec","kwiecien","maj","czerwiec","lipiec","sierpien","wrzesien","pazdziernik","listopad","grudzien",
              "poniedzialek","wtorek","sroda","czwartek","piatek","sobota","niedziela","wiosna","lato","jesien","zima",
              "slonce","deszcz","snieg","grad","pochmurno","slisko","wiatr","gololedz","mgla","cieplo","zimno","burza","piorun","grzmot",
              "zero","jeden","dwa","trzy","cztery","piec","szesc","siedem","osiem","dziewiec","dziesiec",
              "jedenascie","dwanascie","trzynascie","czternascie","pietnascie","szesnascie","siedemnascie","osiemnascie","dziewietnascie","dwadziescia",
              "glosniej","ciszej","do przodu","do tylu","igla","odstaw","postaw","przewin","losuj","wybierz"]

def wav2model(wavepath_list, phrase):
  """
  Zasadniczo to GMM nie przyjmuje MFCC tak od razu, tylko dopiero przy 
  fittowaniu. Więc oddzielna funkcja nie ma sensu? chyba? 
  """

  for i in range(0, len(wavepath_list)):
    nazwa = wavepath_list[i]  # wydobycie nazwy pliku
    fs, data = wav.read(filename=nazwa)  # wczytanie pliku
    macierz_MFCC = compute_mfcc(signal=data, fs=fs)  # mfcc dla jednego pliku
    lista_mfcc.extend(macierz_MFCC)  # dodanie mfccc do listy

  gmm = GaussianMix(8,10)   #nie wiem co z tymi komponentami
  model = gmm.fit(macierz_MFCC)

  return model

def GaussianMix(n,n_iter):
  """
  Zwraca Gaussian Mixture potrzebny do wytrenowania modelu
  """
    gmm=sklearn.mixture.GaussianMixture(n_components=n, 
                                        covariance_type='full', 
                                        tol=0.001, 
                                        reg_covar=1e-06,
                                        max_iter=n_iter, 
                                        n_init=5, 
                                        init_params='kmeans', 
                                        weights_init=None,
                                        means_init=None, 
                                        precisions_init=None, 
                                        random_state=10,
                                        warm_start=False, 
                                        verbose=0, 
                                        verbose_interval=10)
    return gmm

def trainModelGMM(mfcc, gmm_list):
  return gmm_list.fit(mfcc)
