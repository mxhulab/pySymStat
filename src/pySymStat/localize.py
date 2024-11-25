import argparse
import starfile
import sys
import numpy as np
import pandas as pd
from math import *
from pathlib import Path
from rich.progress import track

from .symmetry import Symmetry
from .conversion import euler_to_quat, vec_to_euler, quat_to_euler
from .quaternion import quat_conj, quat_mult, quat_rotate

def parse_args():
    parser = argparse.ArgumentParser(description = 'Locating subparticles considering molecular symmetry.')

    basic = parser.add_argument_group('Basic arguments')
    basic.add_argument('--i',   type = Path,  required = True, help = 'Input particle star file. Only for RELION version >3.1.')
    basic.add_argument('--o',   type = Path,  required = True, help = 'Output subparticle star file.')
    basic.add_argument('--sym', type = str,   default = 'C1',  help = 'Molecular symmetry group.')
    basic.add_argument('--v',   type = float, nargs = 3,       help = 'The vector (in Angstrom) of subparticle')
    basic.add_argument('--eps', type = float, default = -inf,  help = 'The distance threshold (in Angstrom) for identifying different subparticles.')

    extract = parser.add_argument_group('Arguments for re-centering subparticles in cryoSPARC')
    extract.add_argument('--recenter', action = 'store_true', help = 'Recenter the subparticles in micrograph.')
    extract.add_argument('--angpix',   type = float,          help = 'Pixelsize of MICROGRAPH (in Angstrom).')

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def unique(sym, v, eps):
    '''Find the representatives of the orbit.
    '''
    M = sym.size
    elems = sym.elems
    orbits = quat_rotate(elems.T, v).T
    indices = [i for i in range(M) if all(np.linalg.norm(orbits[i] - orbits[j]) > eps for j in range(i))]
    return elems[indices]

def main():
    args = parse_args()
    v = np.array(args.v)
    sym = Symmetry(args.sym)
    elems = unique(sym, v, args.eps)
    print(f'Locate {len(elems)} subparticles in 1 particle.')

    star = starfile.read(args.i, always_dict = True)
    optics, particles = star['optics'], star['particles']
    qs = euler_to_quat(np.array([particles['rlnAngleRot'], particles['rlnAngleTilt'], particles['rlnAnglePsi']]))
    vn = v / np.linalg.norm(v)
    qv = quat_conj(euler_to_quat(np.array([*vec_to_euler(vn), 0])))

    N, M = qs.shape[1], elems.shape[0]
    subparticles = []
    for i in track(range(N), description = 'Generating subparticles ...'):
        qm = qs[:, i]
        for j in range(M):
            particles_ij = particles.loc[i].copy()

            qg = elems[j]
            rot, tilt, psi = quat_to_euler(quat_mult(quat_mult(qm, qg), qv))
            particles_ij['rlnAngleRot'] = rot
            particles_ij['rlnAngleTilt'] = tilt
            particles_ij['rlnAnglePsi'] = psi

            x, y, z = quat_rotate(quat_mult(qm, qg), v)
            particles_ij['rlnOriginXAngst'] -= x
            particles_ij['rlnOriginYAngst'] -= y
            if args.recenter:
                assert 'rlnCoordinateX' and 'rlnCoordinateY' in particles_ij
                assert args.angpix is not None
                angpix = args.angpix
                nx = floor(particles_ij['rlnOriginXAngst'] / angpix)
                ny = floor(particles_ij['rlnOriginYAngst'] / angpix)
                particles_ij['rlnCoordinateX'] -= nx
                particles_ij['rlnCoordinateY'] -= ny
                particles_ij['rlnOriginXAngst'] -= nx * angpix
                particles_ij['rlnOriginYAngst'] -= ny * angpix
            particles_ij['rlnDefocusU'] -= z
            particles_ij['rlnDefocusV'] -= z

            subparticles.append(particles_ij)

    subparticles = pd.DataFrame(subparticles)
    subparticles.set_index(np.arange(N * M), inplace = True)
    starfile.write({ 'optics' : optics, 'particles' : subparticles }, args.o, overwrite = True)
