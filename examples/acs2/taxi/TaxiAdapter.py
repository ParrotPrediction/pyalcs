from lcs.agents import EnvironmentAdapter


class TaxiAdapter(EnvironmentAdapter):

    @staticmethod
    def to_genotype(phenotype):
        """
        Converts environment representation of a state to LCS
        representation.
        """
        phenotype = (str(phenotype), )
        return phenotype

    @staticmethod
    def to_phenotype(genotype):
        """
        Converts LCS representation of a state to environment
        representation.
        """
        return genotype[0]
