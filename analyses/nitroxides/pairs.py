import numpy

class IonPair:
    def __init__(self, q: float, a1: float, a2: float, s1: float = 1.0, s2: float = 1.0):
        self.q = q
        self.a1 = a1
        self.a2 = a2
        
        self.mu = q * (a1 + a2) * s1
        self.a = (a1**3 + a2**3)**(1/3) * s2
    
    @staticmethod
    def _e_born_solvation(q: float, a: float, epsr: float, epsi: float = 1.0) -> float:
        """Solvatation energy from Born theory"""
        return q ** 2 / (2*a) * (1/epsr - 1/epsi)
    
    @staticmethod
    def _e_coulomb(q1: float, q2: float, r: float, eps: float) -> float:
        """Coulombic interaction between two charges in a given medium"""
        return q1 * q2 / (r*eps)
    
    @staticmethod
    def _e_coulomb2(q1: float, q2: float, mu: float, eps: float) -> float:
        """Coulombic interaction / formation of a dipole"""
        return q1 * q2 * numpy.abs(q1) / (eps * mu)
    
    @staticmethod
    def _e_dipole_onsager(mu: float, a: float, epsr: float, epsi: float = 1.0) -> float:
        """Solvatation energy of a dipole in a cavity, according to Onsager"""
        return -3*(epsr-epsi)/((2*epsr+1)*(2*epsi+1))*mu**2/a**3
    
    def e_born_solvation(self, epsr: float, epsi: float = 1.0) -> float:
        return IonPair._e_born_solvation(self.q, self.a1, epsr, epsi) + IonPair._e_born_solvation(-self.q, self.a2, epsr, epsi)
    
    def e_coulomb(self, epsi: float = 1.0) -> float:
        return IonPair._e_coulomb(self.q, -self.q, self.a1 + self.a2, epsi)
    
    def e_coulomb2(self, eps: float) -> float:
        return IonPair._e_coulomb2(self.q, -self.q, self.mu, eps)
    
    def e_dipole_onsager(self, epsr, epsi: float = 1.0) -> float:
        return IonPair._e_dipole_onsager(self.mu, self.a, epsr, epsi)
    
    def e_pair(self, epsr: float, epsi: float = 1.0) -> float:
        return self.e_coulomb2(epsi) + self.e_dipole_onsager(epsr, epsi) - self.e_born_solvation(epsr, epsi)
