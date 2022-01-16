import pandas as pd
from pathlib import Path 
from ruamel.yaml import YAML

class DataLoader():

  def __init__(self, path_to_set):
    path_to_set = Path(path_to_set)
    if path_to_set.exists() and path_to_set.is_dir():
        self.loaded_set = pd.DataFrame({"path": list(path_to_set.glob("**/*.wav"))})
        self.dict_normalize = {
            "ą" : "a",
            "ć" : "c", 
            "ę" : "e", 
            "ł" : "l", 
            "ń" : "n", 
            "ó" : "o", 
            "ś" : "s", 
            "ź" : "z", 
            "ż" : "z"
        }
        self.dict_rename = {
            '0' : 'zero',
            '1' : 'jeden',
            '2' : 'dwa',
            '3' : 'trzy',
            '4' : 'cztery',
            '5' : 'piec',
            '6' : 'szesc',
            '7' : 'siedem',
            '8' : 'osiem',
            '9' : 'dziewiec',
            '10' : 'dziesiec',
            '11' : 'jedenascie',
            '12' : 'dwanascie',
            '13' : 'trzynascie',
            '14' : 'czternascie',
            '15' : 'pietnascie',
            '16' : 'szesnascie',
            '17' : 'siedemnascie',
            '18' : 'osiemnascie',
            '19' : 'dziewietnascie',
            '20' : 'dwadziescia',
            'dotylu' : "do_tylu",
            'doprzodu' : "do_przodu",
            'grudzien2' : "grudzien"
        }
    else:
        raise FileNotFoundError
    print("Parsed path to set")


  def path_to_str(self,path_format):
    return str(path_format)


  def get_set_name(self,path_to_file):
    if "nieznani" in str(path_to_file):
      return "test"
    elif "grupa_ciag_uczacy" in str(path_to_file):
      return "train"
    else:
      return "unknown"


  def get_normalize_name(self,path_to_file):

    name = str(Path(path_to_file).name)[:-4].lower()
    name = name.replace(".wav", "")
    name = name.replace("’", "-")
    name = name.replace("'", "-")

    if "-" in name:
      names = name.split("-")
      if len(names) > 1:
          name = names[1]

    name = name.replace(" ", "_")

    while name[-1] == "_":
      name = name[:-1]

    for n in self.dict_normalize.keys():
        name = name.replace(n, self.dict_normalize[n])

    for n in self.dict_rename.keys():
      if name == n:
        name = self.dict_rename[n]

    
    return name


  def split_to_train_test(self):

    self.loaded_set.path = self.loaded_set.path.apply(self.path_to_str)
    self.loaded_set["command"] = self.loaded_set.path.apply(self.get_normalize_name)

    self.loaded_set["set"] = self.loaded_set.path.apply(self.get_set_name)

    return self.loaded_set[self.loaded_set.set == "train"], self.loaded_set[self.loaded_set.set == "test"]



        


