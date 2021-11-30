import pubchempy

lines = open("pubchem_crawled").readlines()
errors = []
D = {}
for i, line in enumerate(lines):
    content = line.split(',')
    chem_name = content[0].lower()
    smiles = content[1]
    toxicity = content[2].strip()
    D[(chem_name, smiles)] = toxicity

for key in D.keys():
    f = open(f"./molecules/{key[0]}.smi", 'w')
    f.write(key[1])