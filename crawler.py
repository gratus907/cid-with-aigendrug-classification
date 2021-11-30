import pubchempy

def toxicity_class(s):
    if s.startswith('Ambiguous'):
        return 0, False
    if s == "vNo-DILI-Concern":
        return 0, True
    if s == 'vLess-DILI-Concern':
        return 1, True
    if s == 'vMost-DILI-Concern':
        return 1, True
    else: return 0, False
lines = open(f"DILIrank-DILIscore_List.txt").readlines()
errors = []
for i, line in enumerate(lines):
    content = line.split(':')
    chem_name = content[1]
    toxicity, valid = toxicity_class(content[4])
    if not valid: continue
    try:
        compound = pubchempy.get_compounds(chem_name, 'name')[0]
        cid = compound.cid
        smile = compound.to_dict(properties=['isomeric_smiles'])['isomeric_smiles']
        print(f"{chem_name},{smile},{toxicity}")
    except IndexError:
        errors.append((chem_name, toxicity))
