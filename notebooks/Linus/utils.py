import os
from os.path import join
import sys
import yaml
import numpy as np
import torch

PROJECT_PATH = "/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection"
sys.path.append(PROJECT_PATH)
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR


def load_config(path, additional_args={}):
    """
    Lädt die Konfiguration aus der YAML-Datei und merged sie mit einem zusätzlichen Trainingskonfigurationsfile.
    Zusätzliche Argumente überschreiben vorhandene Einträge.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Versuche, das Trainingskonfigurationsfile zu laden
    try:
        with open('./training/config/train_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser('~/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml'), 'r') as f:
            config2 = yaml.safe_load(f)
    
    # Wenn im ersten Konfigurationsfile ein label_dict vorhanden ist, aktualisiere das Trainingsconfig.
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    
    config.update(config2)
    
    if config.get('dry_run', False):
        config['nEpochs'] = 0
        config['save_feat'] = False
        
    # Überschreibe mit zusätzlichen Argumenten.
    for key, value in additional_args.items():
        config[key] = value
    return config

def load_model(config):
    """
    Lädt und gibt das Modell zurück, basierend auf der im Config angegebenen Modelldefinition.
    """
    print("Registered models:", DETECTOR.data.keys())
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    return model

###############################################
# Einfacher Trainer, der die nötigen Attribute liefert.
###############################################

class Trainer:
    def __init__(self, model, save_path, epoch):
        self.model = model
        self.save_path = save_path
        self.epoch = epoch

###############################################
# Basisklasse Analyser (wie gewünscht)
###############################################

class Analyser:
    default_config = {}

    def __init__(self, trainer, **config):
        self.trainer = trainer
        # Setze Standardwerte aus default_config, falls nicht überschrieben.
        for k, v in self.default_config.items():
            if k not in config:
                config[k] = v
        self.config = config
        self.results = None

    def analysis(self):
        raise NotImplementedError("Need to implement analysis function.")

    def run(self):
        results = self.analysis()
        self.save_results(results)

    def save_results(self, results):
        save_path = join(self.trainer.save_path,  self.get_save_folder())
        os.makedirs(save_path, exist_ok=True)
        for k, v in results.items():
            np.savetxt(join(save_path, "{}.np".format(k)), v)

        with open(join(save_path, "config.log"), "w") as file:
            for k, v in self.get_config().items():
                k_v_str = "{k}: {v}".format(k=k, v=v)
                print(k_v_str)
                file.writelines([k_v_str, "\n"])

    def get_save_folder(self, epoch=None):
        raise NotImplementedError("Need to implement get_save_folder function.")

    def get_config(self):
        config = self.config
        config.update({"epoch": self.trainer.epoch})
        return config

    def load_results(self, epoch=None):
        save_path = join(self.trainer.save_path,  self.get_save_folder(epoch))
        if not os.path.exists(save_path):
            return
        results = dict()
        files = [f for f in os.listdir(save_path) if f.endswith(".np")]
        for file in files:
            results[file[:-3]] = np.loadtxt(join(save_path, file))
        self.results = results


class Analyser:

    default_config = {}

    def __init__(self, trainer, **config):
        self.trainer = trainer
        for k, v in self.default_config.items():
            if k not in config:
                config[k] = v
        self.config = config
        self.results = None

    def analysis(self):
        raise NotImplementedError("Need to implement analysis function.")

    def run(self):
        results = self.analysis()
        self.save_results(results)

    def save_results(self, results):
        save_path = join(self.trainer.save_path,  self.get_save_folder())
        os.makedirs(save_path, exist_ok=True)
        for k, v in results.items():
            np.savetxt(join(save_path, "{}.np".format(k)), v)

        with open(join(save_path, "config.log"), "w") as file:
            for k, v in self.get_config().items():
                k_v_str = "{k}: {v}".format(k=k, v=v)
                print(k_v_str)
                file.writelines([k_v_str, "\n"])

    def get_save_folder(self, epoch=None):
        raise NotImplementedError("Need to implement get_save_folder function.")

    def get_config(self):
        config = self.config
        config.update({"epoch": self.trainer.epoch})
        return config

    def load_results(self, epoch=None):
        save_path = join(self.trainer.save_path,  self.get_save_folder(epoch))
        # print("Trying to load results from", save_path)
        if not os.path.exists(save_path):
            return
        results = dict()
        files = [f for f in os.listdir(save_path) if f.endswith(".np")]
        for file in files:
            results[file[:-3]] = np.loadtxt(join(save_path, file))
        self.results = results