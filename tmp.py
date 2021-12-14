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

class Chemical:
    def __init__(self, name, smiles, fp, toxicity):
        self.name = name
        self.smiles = smiles
        self.fp = fp
        self.toxicity = toxicity

    def __str__(self):
        return f"CHEMICAL {self.name} {self.smiles} : {self.toxicity} {self.fp}"

def read_files(rewrite_fp = False):
    if rewrite_fp:
        fingerprint_file = open("./data/tingli_fingerprint.csv").readlines()
    else:
        fingerprint_file = open("./data/tingli_full_fingerprint.csv").readlines()
    chemical_dict = dict()
    for line in fingerprint_file:
        content = line.split(',')
        chem_name = content[0].replace('"','').replace('AUTOGEN_', '')
        fp = [int(x) for x in list("".join(content[1:]).strip())]
        chemical_dict[chem_name] = Chemical(chem_name, "", fp, "")

    train_test = open(f"./data/train_idx.csv").readlines()
    train_chem_names = set([line.split(',')[0] for line in train_test])

    chemical_info = open("./data/tingli-smiles").readlines()
    for line in chemical_info:
        content = line.split(',')
        chem_name = content[0].replace('"','')
        smiles = content[1]
        toxicity = int(content[2])
        chemical_dict[chem_name].smiles = smiles
        chemical_dict[chem_name].toxicity = toxicity

    if rewrite_fp:
        query_smiles = parse_input("./data/query_smiles")
        full_fp = open("./data/tingli_full_fingerprint.csv", 'w')
        count_match = 0
        itr = 1; total_chems = len(chemical_dict.values())
        for chemical in chemical_dict.values():
            m = Chem.MolFromSmiles(chemical.smiles, sanitize=False)
            for query_smile in query_smiles:
                query_mol = Chem.MolFromSmiles(query_smile, sanitize=False)
                b = 1 if m.HasSubstructMatch(query_mol) else 0
                count_match += b
                chemical.fp.append(b)
            train_or_test = 'train' if chemical.name in train_chem_names else 'test'
            full_fp.write(f'"{chemical.name}",{train_or_test},{chemical.toxicity},{",".join([str(elem) for elem in chemical.fp])}\n')
            print(f"{itr}/{total_chems} processed, found {count_match}"); itr+=1
    return list(chemical_dict.values())


#print(train_chem_names)
read_files(True)