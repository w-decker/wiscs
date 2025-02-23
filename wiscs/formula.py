import re

class RawFormula(object):
    def __init__(self, formula: str):
        self.raw = formula
        self.__post_init__()

    def __post_init__(self):
        def is_formula(s):
            return re.fullmatch(
                r'\(\s*[^()]+\s*\|\s*[^()]+\s*\)(?:\s*\+\s*\(\s*[^()]+\s*\|\s*[^()]+\s*\))*', s
            ) is not None

        if not is_formula(self.raw):
            raise ValueError("Invalid formula")
        self.formula = self.raw

class Formula(RawFormula):

    def __init__(self, formula:str):
        super().__init__(formula)

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        return f"{self.formula}"
    
    def __iter__(self):
        pattern = r'\(\s*[^()]+\s*\|\s*[^()]+\s*\)'
        for match in re.finditer(pattern, self.formula):
            yield match.group().strip()

    def __len__(self):
        return len(list(self.__iter__()))
    
    def __getitem__(self, index):
        return list(self.__iter__())[index]