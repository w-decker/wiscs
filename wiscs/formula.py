import re

class Formula(object):

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
    
    def __delitem__(self, index):
        items = list(self.__iter__())
        if index < 0 or index >= len(items):
            raise IndexError("Index out of range")
        del items[index]
        self.formula = " + ".join(items)
        
    def __add__(self, other):
        if isinstance(other, str):
            return Formula(self.formula + " + " + other)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, str):
            items = list(self.__iter__()) 
            if other in items:
                items.remove(other)
            self.formula = " + ".join(items)
            return Formula(" + ".join(items))
        return NotImplemented
    
    def __str__(self):
        return self.formula