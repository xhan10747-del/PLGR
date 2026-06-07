import re
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class SMILESFixer:
    def __init__(self):
        self.fix_rules = [
            (r'\[P\]\(=O\)\(=O\)=O', 'P(=O)(O)O'),
            (r'\[P\]\(=O\)\(=O\)\(=O\)', 'P(=O)(O)O'),
            (r'P\(=O\)\(=O\)=O', 'P(=O)(O)O'),
            (r'=O\)=O', '=O)O'),
            (r'\[P\]\(=O\)\(=O\)OP', 'P(=O)(O)OP'),
            (r'\[P\]\(=O\)\(=O\)OC', 'P(=O)(O)OC'),
            (r'\[P\]\(=O\)\(=O\)\(=O\)OP\(=O\)\(=O\)OP\(=O\)\(=O\)', 'P(=O)(O)OP(=O)(O)OP(=O)(O)'),
            (r'P\(=O\)\(=O\)O\[P\]\(=O\)\(=O\)=O', 'P(=O)(O)OP(=O)(O)O'),
            (r'\[P\]\(=O\)\(=O\)=O', 'P(=O)(O)O'),
            (r'\[N\+\]', '[N+]'),
            (r'\[NH\+\]', '[NH+]'),
            (r'\[NH3\+\]', '[NH3+]'),
            (r'\)\)', ')'),
            (r'\(\(', '('),
        ]
    def fix_brackets(self, smiles):
        open_brackets = smiles.count('(')
        close_brackets = smiles.count(')')
        if open_brackets > close_brackets:
            smiles += ')' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            excess = close_brackets - open_brackets
            for _ in range(excess):
                last_close = smiles.rfind(')')
                if last_close != -1:
                    smiles = smiles[:last_close] + smiles[last_close+1:]
        return smiles
    def apply_fix_rules(self, smiles):
        for pattern, replacement in self.fix_rules:
            smiles = re.sub(pattern, replacement, smiles)
        return smiles
    def validate_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    def fix_smiles(self, smiles):
        if not smiles or not isinstance(smiles, str):
            return None
        original_smiles = smiles
        smiles = self.apply_fix_rules(smiles)
        smiles = self.fix_brackets(smiles)
        if self.validate_smiles(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                logger.debug(f"Successfully fixed: {original_smiles[:50]}... -> {canonical_smiles[:50]}...")
                return canonical_smiles
            except:
                logger.warning(f"Failed to canonicalize: {smiles[:50]}...")
                return smiles
        return self.aggressive_fix(original_smiles)
    def aggressive_fix(self, smiles):
        try:
            phosphate_chain_patterns = [
                (r'\[P\]\(=O\)\(=O\)\(=O\)OP\(=O\)\(=O\)OP\(=O\)\(=O\)', 'P(=O)(O)O'),
                (r'P\(=O\)\(=O\)O\[P\]\(=O\)\(=O\)=O', 'P(=O)(O)O'),
                (r'\[P\]\(=O\)\(=O\)\(=O\)', 'P(=O)(O)O'),
            ]
            for pattern, replacement in phosphate_chain_patterns:
                smiles = re.sub(pattern, replacement, smiles)
            problematic_patterns = [
                r'=O\)=O\)',
                r'\)=O\)=O',
                r'\(/C\)/C',
            ]
            for pattern in problematic_patterns:
                smiles = re.sub(pattern, '', smiles)
            smiles = self.fix_brackets(smiles)
            if self.validate_smiles(smiles):
                mol = Chem.MolFromSmiles(smiles)
                return Chem.MolToSmiles(mol, canonical=True)
            no_phosphate = re.sub(r'P\([^)]*\)', '', smiles)
            no_phosphate = re.sub(r'\[P\][^\]]*', '', no_phosphate)
            if self.validate_smiles(no_phosphate):
                mol = Chem.MolFromSmiles(no_phosphate)
                logger.warning(f"Removed all phosphate groups to fix: {smiles[:50]}...")
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.error(f"Aggressive fix failed for {smiles[:50]}...: {e}")
        return None
    def batch_fix_smiles(self, smiles_list):
        results = []
        fixed_count = 0
        failed_count = 0
        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                logger.info(f"Processing {i}/{len(smiles_list)} SMILES...")
            fixed_smiles = self.fix_smiles(smiles)
            if fixed_smiles:
                results.append(fixed_smiles)
                if fixed_smiles != smiles:
                    fixed_count += 1
            else:
                results.append(None)
                failed_count += 1
        logger.info(f"Batch processing complete: {fixed_count} fixed, {failed_count} failed")
        return results
def test_fixer():
    fixer = SMILESFixer()
    test_cases = [
        "[C@@H]1(O[P](=O)(=O)=O)[C@H](O)[C@H](O[P](=O)(=O)=O)[C@@H](O[P](=O)(=O)=O)[C@H](O[P](=O)(=O)=O)[C@H]1O)",
        "[P](=O)(=O)(=O)OP(=O)(=O)OP(=O)(=O)OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c(N)ncnc12",
        "CC(=CCC/C(=C/CC/C(=C/CC/C(=C/COP(=O)(=O)O[P](=O)(=O)=O)/C)/C)/C)C",
    ]
    print("Testing SMILES Fixer:")
    for i, smiles in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"Original: {smiles}")
        fixed = fixer.fix_smiles(smiles)
        print(f"Fixed:    {fixed}")
        print(f"Valid:    {fixer.validate_smiles(fixed) if fixed else False}")
if __name__ == "__main__":
    test_fixer()
