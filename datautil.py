import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.transforms import transforms
from rdkit import Chem
import networkx as nx
import sys, io
import csv

def parse_input(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    smiles = []
    for line in lines:
        smile = line.strip()
        if len(smile) == 0: continue
        smiles.append(smile)
    return smiles

def read_smiles(smiles_path):
    smiles = parse_input(smiles_path)
    mols = dict()
    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smile, sanitize=False)
            mols[smile] = mol
        except AttributeError as e:
            pass
    return smiles, mols

bayesian_weight = np.ones(3501)

class Chemical:
    def __init__(self, name, smiles, fp, toxicity):
        self.name = name
        self.smiles = smiles
        self.fp = fp
        self.toxicity = toxicity
        self.mol = Chem.MolFromSmiles(self.smiles, sanitize=False)

    def __str__(self):
        return f"CHEMICAL {self.name} {self.smiles} : {self.toxicity} {self.fp}"

def read_files():
    fingerprint_file = open("./data/fingerprint.csv").readlines()
    chemical_dict = dict()
    for line in fingerprint_file:
        content = line.split(',')
        chem_name = content[0].replace('"','')
        fp = [int(x) for x in list("".join(content[1:]).strip())]
        chemical_dict[chem_name] = Chemical(chem_name, "", fp, "")

    chemical_info = open("./data/smiles").readlines()
    for line in chemical_info:
        content = line.split(',')
        chem_name = content[0].replace('"','')
        smiles = content[1]
        toxicity = int(content[2].strip())
        chemical_dict[chem_name].smiles = smiles
        chemical_dict[chem_name].toxicity = toxicity

    query_smiles, query_mols = read_smiles("./data/query_smiles")
    full_fp = open("./data/full_fingerprint.csv", 'w')
    count_match = 0
    itr = 0; total_chems = len(chemical_dict.values())
    for chemical in chemical_dict.values():
        print(chemical.mol)
        for query_smile in query_smiles:

            count_match += int(chemical.mol.HasSubstructMatch(query_mols[query_smile]))
            chemical.fp.append(int(chemical.mol.HasSubstructMatch(query_mols[query_smile])))
        #full_fp.write(f'"{chemical.name}",{"".join([str(elem) for elem in chemical.fp])}\n')
        print(f"{itr}/{total_chems} processed, found {count_match}"); itr+=1
    return list(chemical_dict.values())

def read_aigen_file():
    query_smiles, query_mols = read_smiles("./data/query_smiles")
    count_match = 0
    f = open("./data/aigendrug_dili").readlines()
    chemical_dict = dict()
    idx = 0
    for line in f:
        content = line.split('	')[1:]
        chem_name = f"Aigen_{idx}"; idx += 1
        smiles = content[0].replace('"', '')
        toxicity = int(float(content[1].strip()))
        fp = []
        chem = Chem.MolFromSmiles(smiles, sanitize=False)
        for q in query_smiles:
            b = chem.HasSubstructMatch(Chem.MolFromSmiles(q, sanitize=False))
            b = int(b)
            count_match += b
            fp.append(b)
        chemical_dict[chem_name] = Chemical(name=chem_name, smiles=smiles, fp=fp, toxicity=toxicity)
        chemical_dict[chem_name].smiles = smiles
        chemical_dict[chem_name].toxicity = toxicity
    return list(chemical_dict.values())

class ChemicalDILIDataset(Dataset):
    def __init__(self, chem, use_torch_tensor = False, weight=True):
        self.chemicals = chem
        self.use_torch_tensor = use_torch_tensor
        self.weight = weight

    def __len__(self):
        return len(self.chemicals)

    def __getitem__(self, idx):
        if self.weight:
            fp = np.array(self.chemicals[idx].fp) * (bayesian_weight)
            fp = (fp - fp.min()) / (fp.max() - fp.min())
        else: fp = np.array(self.chemicals[idx].fp)
        tox = np.array([self.chemicals[idx].toxicity])
        if self.use_torch_tensor:
            fp = torch.Tensor(fp)
            tox = torch.Tensor(tox)
        return fp, tox

def load_DILI_data():
    chemicals = read_aigen_file()
    chem_train, chem_test = train_test_split(chemicals, test_size=0.1)
    train_set = ChemicalDILIDataset(chem_train, use_torch_tensor = True, weight = False)
    test_set = ChemicalDILIDataset(chem_test, use_torch_tensor = True, weight = False)
    train_loader = DataLoader(train_set, batch_size=1)
    test_loader = DataLoader(test_set)
    return train_loader, test_loader

def read_data_as_np(weight = True):
    chemicals = read_aigen_file()
    chem_train, chem_test = train_test_split(chemicals, test_size=0.2)
    train_set = ChemicalDILIDataset(chem_train, use_torch_tensor=False, weight=weight)
    test_set = ChemicalDILIDataset(chem_test, use_torch_tensor=False, weight=weight)
    return train_set, test_set