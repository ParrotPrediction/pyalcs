from lcs.agents import EnvironmentAdapter


class TaxiAdapter(EnvironmentAdapter):

    @classmethod
    def to_genotype(cls, phenotype):
        """
        Converts environment representation of a state to LCS
        representation.
        """
        phenotype = (str(phenotype), )
        return phenotype

    @classmethod
    def to_phenotype(cls, genotype):
        """
        Converts LCS representation of a state to environment
        representation.
        """
        return genotype[0]
