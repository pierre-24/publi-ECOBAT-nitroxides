import numpy
import pathlib
import pandas
from scipy.spatial import distance_matrix

# -- Debye-Huckel theory

AU_TO_M = 5.291772e-11  # m
KB = 3.166811563e-6  # Eh / K
AVOGADRO = 6.02214076e23  # mol⁻¹


def kappa2(C: float, z: int, eps_r: float, T: float = 298.15) -> float:
    # n = N / V
    n = C * AVOGADRO * AU_TO_M ** 3 * 1e3  # in bohr⁻³
    
    # k² = n*q² / eps0 * epsr * kB * T =  n * (z * e0)² * (1 / 4 pi eps0) * [4 pi / epsr * kB * T]
    k2 = 4 * numpy.pi / (eps_r * KB * T) * n * z**2  # in bohr⁻¹
    
    return k2

def G_DH(z: float, kappa: float, epsilon_r: float, a: float) -> float:
    G = -27.212 * z **2 / epsilon_r * kappa / (kappa * a) ** 3 * (numpy.log(1 +  kappa * a) - kappa * a + .5 * (kappa * a) ** 2)  # in eV
    G[kappa < 1e-5] = 0
    
    return G

def dG_DH(z_reac: int, z_prod: int, a_reac: float, a_prod: float, epsilon_r: float, c_elt: float, z_elt: int = 1, c_act: float = 0.1):
    # Note: assume charge compensation
    kappa_reac = numpy.sqrt(kappa2(c_act, z_reac, epsilon_r) + kappa2(c_act, -z_reac, epsilon_r) + kappa2(c_elt, z_elt, epsilon_r) + kappa2(c_elt, -z_elt, epsilon_r))  # in bohr⁻¹
    kappa_prod = numpy.sqrt(kappa2(c_act, z_prod, epsilon_r) + kappa2(c_act, -z_prod, epsilon_r) + kappa2(c_elt, z_elt, epsilon_r) + kappa2(c_elt, -z_elt, epsilon_r))  # in bohr⁻¹
    
    dG = G_DH(z_prod, kappa_prod, epsilon_r, a_prod) - G_DH(z_reac, kappa_reac, epsilon_r, a_reac)
    
    return dG
    

# -- Position labels on graph (using a monte-carlo metropolis approach)
class LabelPositioner:
    def __init__(
        self, 
        points_position: numpy.ndarray, 
        labels: list,
        labels_position: numpy.ndarray,
        labels_kwargs: list = None,
    ):
        self.points_position = points_position
        self.labels = labels
        self.labels_position = labels_position
        self.labels_kwargs = labels_kwargs if labels_kwargs is not None else [{}] * len(self.labels)
        
    @classmethod
    def from_file(cls, path: pathlib.Path, points_position: numpy.ndarray, labels: list = None, **kwargs):
        if path.exists():
            data = pandas.read_csv(path)
            labels = data['label']
            labels_position = numpy.vstack([data['x'], data['y']]).T
        else:
            labels_position = points_position.copy()
        
        return cls(points_position, labels, labels_position, **kwargs)
    
    def optimize(
        self, 
        N: int = 1000, 
        dx: float = .001, 
        beta: float=1e6,
        scale: tuple = (1., 1.),
        krep: float=1, 
        kspring: float=100, 
        b0: float=0.015,
        c: float=0.05, 
    ):
        def _E():
            dinter = distance_matrix(self.labels_position * scale, self.labels_position * scale)
            dp = distance_matrix(self.labels_position * scale, self.points_position * scale)
            E = 0
            
            ninter = dinter[numpy.triu_indices(len(self.labels), k=1)]
            np = dp[numpy.triu_indices(len(self.labels), k=1)]
            
            dists = numpy.hstack([ninter[ninter < c], np[np < c]])
            E += numpy.sum(4*krep * ((1.5 * b0 / dists) ** 12) * (c-dists))
            E += numpy.sum(kspring*(dp[numpy.diag_indices(len(self.labels))] - b0)**2)
            
            return E
        
        E = _E()
        nacc = 0
        
        for i in range(N):
            for j in range(len(self.labels)):
                old_pos = self.labels_position[j].copy()
                self.labels_position[j] += (numpy.random.rand(2) - .5)*dx
                nE = _E()
                if numpy.random.rand() < min(1, numpy.exp(beta*(E - nE))):
                    E = nE
                    nacc += 1
                else:
                    self.labels_position[j] = old_pos
            
            print(i, E, '{:.3f} {}'.format(nacc / ((i+1) * N)*100, nacc))
    
    def add_labels(self, ax):
        for i in range(len(self.labels)):
            ax.text(*self.labels_position[i], self.labels[i], **self.labels_kwargs[i])
    
    def save(self, path: pathlib.Path):
        data = {'label': self.labels, 'x': self.labels_position[:, 0], 'y': self.labels_position[:, 1]}
        pandas.DataFrame(data=data).to_csv(path)
    
