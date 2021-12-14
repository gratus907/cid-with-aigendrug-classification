import pubchempy

crawled = open(f"./pubchem_crawled").readlines()
chem_data = dict()
for i, line in enumerate(crawled):
    content = line.split(',')
    chem_name = content[0].upper()
    chem_smiles = content[1].strip()
    chem_data[chem_name] = chem_smiles

lines = open(f"./tingli_data.csv").readlines()
errors = []
idx=1
f = open(f"./tingli-smiles", 'w')
for i, line in enumerate(lines):
    content = line.split(',')
    chem_name = content[1]
    toxicity = int(content[2])
    if chem_name in chem_data.keys():
        smile = chem_data[chem_name]    
        print(f"Found(pdict) [{idx}] {chem_name}-{smile}-{'TOXIC' if toxicity==1 else 'NONTOXIC'}")
    else:
        try:
            compound = pubchempy.get_compounds(chem_name, 'name')[0]
            cid = compound.cid
            smile = compound.to_dict(properties=['isomeric_smiles'])['isomeric_smiles']    
            print(f"Found(pchem) [{idx}] {chem_name}-{smile}-{'TOXIC' if toxicity==1 else 'NONTOXIC'}")
        except IndexError:
            errors.append((chem_name, toxicity))
            continue 
    idx += 1
    f.write(f"{chem_name},{smile},{toxicity}\n")
    molf = open(f"./molecules-tingli/{chem_name}.smi", 'w')
    molf.write(f"{smile}")

print(f"{len(errors)} errors from pubchem")