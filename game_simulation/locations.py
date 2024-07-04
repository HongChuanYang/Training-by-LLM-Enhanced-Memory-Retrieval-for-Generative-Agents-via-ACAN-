class Location:
    """
    A class to represent a location in the simulated environment.

    Attributes:
    ----------
    name : str
        The name of the location.
    description : str
        A brief description of the location.

    Methods:
    -------
    describe():
        Prints the description of the location.
    """

    def __init__(self, name, description):
        self.name = name
        self.description = description['description']
        self.type = description['type']
        self.agents = description['agents']
    
    def __repr__(self):
        return f"Location({self.name}, {self.description})"
    
    def describe(self):
        return self.description
