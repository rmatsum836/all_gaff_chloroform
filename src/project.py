from flow import FlowProject
import signac
import flow
import matplotlib.pyplot as plt
import mbuild as mb
import mdtraj as md 
import unyt as u
from scipy import stats
import numpy as np
import pickle
from foyer import Forcefield
from scipy.optimize import curve_fit
from get_mol2 import GetSolv, GetIL
from util.decorators import job_chdir
from pkg_resources import resource_filename
from mtools.gromacs.gromacs import make_comtrj
from mtools.post_process import calc_msd
from ramtools.transport.calc_transport import calc_conductivity
from mtools.post_process import calc_density
import os
import environment
import itertools as it
import parmed as pmd
import gafffoyer
import antefoyer
from simtk.unit import *


def workspace_command(cmd):
    """Simple command to always go to the workspace directory"""
    return ' && '.join([
        'cd {job.ws}',
        cmd if not isinstance(cmd, list) else ' && '.join(cmd),
        'cd ..',
    ])


def _run_overall(trj, mol):
     D, MSD, x_fit, y_fit = calc_msd(trj)
     return D, MSD

 
def _save_overall(job, mol, trj, MSD):
        np.savetxt(os.path.join(job.workspace(), 'msd-{}-overall_2.txt'.format    (mol)),
                        np.transpose(np.vstack([trj.time, MSD])),
                                header='# Time (ps)\tMSD (nm^2)')

        fig, ax = plt.subplots()
        ax.plot(trj.time, MSD)
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('MSD (nm^2)')
        fig.savefig(os.path.join(job.workspace(),
                    'msd-{}-overall_2.pdf'.format(mol)))


def _run_multiple(trj, mol):
    D_pop = list()
    #for start_frame in np.linspace(0, 10001, num=200, dtype=np.int):
    for start_frame in np.linspace(0, 10001, num=50, dtype=np.int):
        end_frame = start_frame + 5000
        if end_frame < 10001:
            chunk = trj[start_frame:end_frame]
            print('\t\t\t...frame {} to {}'.format(start_frame, end_frame))
            try:
                D_pop.append(calc_msd(chunk)[0])
            except TypeError:
                import pdb
                pdb.set_trace()
        else:
            continue
    D_bar = np.mean(D_pop)
    D_std = np.std(D_pop)
    return D_bar, D_std


init_file = 'gaff.gro'
em_file = 'em.gro'
nvt_file = 'nvt.gro'
npt_file = 'npt.gro'
sample_file = 'sample.gro'
unwrapped_file = 'sample_unwrapped.xtc'
msd_file = 'msd-all-overall_2.txt'
pair_file = 'direct-matrices-cation-anion.pkl.gz'
pair_fit_file = 'matrix-pairs-solvent-anion.txt'
tau_file = 'tau.txt'
rdf_file = 'rdf-chlor-cation.txt'
all_directs_file = 'all-directs-solvent-cation.pkl.gz'
all_indirects_file = 'all-indirects-solvent-cation.pkl'
cn_file = 'cn-cation-anion-2.txt'

class Project(FlowProject):
    pass

@Project.label
def initialized(job):
    return job.isfile(init_file)

@Project.label
def minimized(job):
    return job.isfile(em_file)

@Project.label
def nvt_equilibrated(job):
    return job.isfile(nvt_file)

@Project.label
def npt_equilibrated(job):
    return job.isfile(npt_file)

@Project.label
def sampled(job):
    return job.isfile(sample_file)

@Project.label
def prepared(job):
    return job.isfile(unwrapped_file)

@Project.label
def msd_done(job):
    return job.isfile(msd_file)

@Project.label
def pair_done(job):
    return job.isfile(pair_file)

@Project.label
def pair_fit_done(job):
    return job.isfile(pair_fit_file)

@Project.label
def directs_done(job):
    return job.isfile(all_directs_file)

@Project.label
def indirects_done(job):
    return job.isfile(all_indirects_file)

@Project.label
def tau_done(job):
    return job.isfile(tau_file)

@Project.label
def rdf_done(job):
    return job.isfile(rdf_file)

#@Project.label
#def cn_done(job):
#    return job.isfile(cn_file)

@Project.operation
@flow.cmd
def plot_energies(job):
    cmd = 'echo 10 0 | gmx energy -f sample.edr'
    return workspace_command(cmd)
    

@Project.operation
@Project.post.isfile(init_file)
def initialize(job):
    with job:
        print(job.get_id())
        print("Setting up packing ...")
        n_acn = job.statepoint()['n_acn']
        acn_conc = job.statepoint()['acn_conc']
        chlor_conc = job.statepoint()['chlor_conc']
        il_conc = job.statepoint()['il_conc']

        if acn_conc == 2 and chlor_conc == 3 and il_conc == 0.66:
            pack_dim = 6 # nm
            sys_dim = 6.5 # nm
        elif acn_conc == 1 and chlor_conc == 3 and il_conc == 0.33:
            pack_dim = 6 # nm
            sys_dim = 6.5 # nm
        else:
            pack_dim = 7 # nm
            sys_dim = 7.5 # nm
        packing_box = mb.Box([pack_dim,pack_dim,pack_dim])
        system_box = mb.Box([sys_dim,sys_dim,sys_dim])
        n_chlor = int(round((n_acn * chlor_conc) / acn_conc))
        n_ions = int(round((n_acn * il_conc) / acn_conc)) 

        # Load in mol2 files as mb.Compound
        cation = GetIL('li')
        cation.name = 'li'
        anion = GetIL(job.statepoint()['anion'])
        anion.name = job.statepoint()['anion']
        acn = GetSolv('ch3cn_gaff')
        acn.name = 'ch3cn'
        chloroform = GetSolv('chlor')
        chloroform.name = 'chlor'
        
        amber = '/raid6/homes/firstcenter/gaff_chloroform/src/'\
                'util/lib/amber_ions.xml'
        amber = Forcefield(amber)

        if n_ions == 0: 
           system = mb.fill_box(compound=[acn, chloroform],
                   n_compounds=[n_acn,n_chlor],
                   box=packing_box)
        elif n_chlor == 0:
           system = mb.fill_box(compound=[acn, cation, anion],
                   n_compounds=[n_acn,n_ions,n_ions],
                   box=packing_box)
        else:
            system = mb.fill_box(compound=[acn, chloroform, cation, anion],
                    n_compounds=[n_acn,n_chlor,n_ions,n_ions],
                    box=packing_box)
        
        cation = mb.Compound()
        anion = mb.Compound()
        acn_cmp = mb.Compound()
        chlor_cmp = mb.Compound()
        for child in system.children:
            if child.name == 'li':
                cation.add(mb.clone(child))
            elif child.name == 'tf2n':
                anion.add(mb.clone(child))
            elif child.name == 'ch3cn':
                acn_cmp.add(mb.clone(child))
            elif child.name == 'chlor':
                chlor_cmp.add(mb.clone(child))

        # Initialize gaff force field 
        gaff = gafffoyer.gafffoyer.load_GAFF()

        if n_chlor > 0:
            chlor_param = gaff.apply(chlor_cmp, residues='chlor', assert_dihedral_params=False, combining_rule='lorentz')
            chlor_param.box = [sys_dim*10,sys_dim*10,sys_dim*10,90,90,90]
            for atom in chlor_param:
                if atom.name.strip() == 'C':
                    atom.charge = -0.3426
                elif atom.name.strip() == 'H':
                    atom.charge = 0.3141
                elif atom.name.strip() == 'Cl':
                    atom.charge = 0.0095
            chlor_charge = chlor_param
            #chlor_charge.residues[0].name = 'chlor'

        acn_param = gaff.apply(acn_cmp, residues='ch3cn', assert_dihedral_params=False, combining_rule='lorentz')
        acn_param.box = [sys_dim*10,sys_dim*10,sys_dim*10,90,90,90]
        # Set C to N bond to a triple bond ordering
        for bond in acn_param.bonds:
            if set([bond.atom1.name,bond.atom2.name]) == set(['C1', 'N']):
                bond.order = 3.0
        for atom in acn_param:
            if atom.name == 'C1':
                atom.charge = 0.392357
            if atom.name == 'C3':
                atom.charge = -0.254298
            elif atom.name.strip()[0] == 'H':
                atom.charge = 0.118465
            elif atom.name.strip()[0] == 'N':
                atom.charge = -0.493454
        acn_charge = acn_param
        #acn_charge.residues[0].name = 'ch3cn'

        # Apply force field to ions
        liPM = amber.apply(cation, residues='li', combining_rule='lorentz')
        anionPM = gaff.apply(anion, residues='tf2n', assert_dihedral_params=False,
            assert_angle_params=False, combining_rule='lorentz')

        # Apply RESP charges to anion
        for atom in anionPM:
            if atom.name.strip()[0] == 'C':
                atom.charge = 0.2526
            if atom.name.strip()[0] == 'F':
                atom.charge = -.1167
            if atom.name.strip()[0] == 'S':
                atom.charge = 1.1271
            if atom.name.strip()[0] == 'O':
                atom.charge = -0.5784
            if atom.name.strip()[0] == 'N':
                atom.charge = -0.7456

        if n_ions == 0:  
            #structure = (acn_charge * n_acn) + (chlor_charge * n_chlor)
            structure = acn_charge + chlor_charge
        elif n_chlor == 0:
            #structure = (acn_charge * n_acn) + li_comp + anion_comp
            structure = acn_charge + liPM + anionPM
        else:
            #structure = (acn_charge * n_acn) + (chlor_charge * n_chlor) + li_comp + anion_comp
            structure = acn_charge + chlor_charge + liPM + anionPM

        # Set 1-4 coulomb scaling to 0.833333
        for adjust in structure.adjusts:
            if adjust.type.chgscale != 0.8333:
                adjust.type.chgscale = 0.8333
 
        for atom in structure.atoms:
            if atom.residue.name in ['li', 'tf2n']:
                atom.charge *= 0.8
        
        print("Saving .gro, .pdb and .top ... ")
        structure.save('gaff.gro', combine='all', overwrite=True)
        structure.save('gaff.top', combine='all', overwrite=True)


@Project.operation
@Project.pre.isfile(init_file)
@Project.post.isfile(em_file)
@flow.cmd
def em(job):
    return _gromacs_str('em', 'init', 'init', job)


@Project.operation
@Project.pre.isfile(em_file)
@Project.post.isfile(nvt_file)
@flow.cmd
def nvt(job):
    return _gromacs_str('nvt', 'em', 'init', job)


@Project.operation
@Project.pre.isfile(nvt_file)
@Project.post.isfile(npt_file)
@flow.cmd
def npt(job):
    return _gromacs_str('npt', 'nvt', 'init', job)


@Project.operation
@Project.pre.isfile(npt_file)
@Project.post.isfile(sample_file)
@flow.cmd
def sample(job):
    return _gromacs_str('sample', 'npt', 'init', job)

@Project.operation
@Project.pre.isfile(sample_file)
@Project.post.isfile(unwrapped_file)
def prepare(job):
    #if job.get_id() == '41fd6198b7f5675f9ecd034ce7c5af73':
    #    pass
    #else:
    trr_file = os.path.join(job.workspace(), 'sample.trr')
    xtc_file = os.path.join(job.workspace(), 'sample.xtc')
    gro_file = os.path.join(job.workspace(), 'sample.gro')
    tpr_file = os.path.join(job.workspace(), 'sample.tpr')
    if os.path.isfile(xtc_file) and os.path.isfile(gro_file):
        unwrapped_trj = os.path.join(job.workspace(),
        'sample_unwrapped.xtc')
        #if not os.path.isfile(unwrapped_trj):
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc nojump'.format(xtc_file, unwrapped_trj, tpr_file))
        res_trj = os.path.join(job.ws, 'sample_res.xtc')
        com_trj = os.path.join(job.ws, 'sample_com.xtc')
        unwrapped_com_trj = os.path.join(job.ws,'sample_com_unwrapped.xtc')
        #if not os.path.isfile(res_trj):
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc res'.format(
                xtc_file, res_trj, tpr_file))
        #if os.path.isfile(res_trj) and not os.path.isfile(com_trj):
        trj = md.load(res_trj, top=gro_file)
        comtrj = make_comtrj(trj)
        comtrj.save_xtc(com_trj)
        comtrj[-1].save_gro(os.path.join(job.workspace(),
            'com.gro'))
        print('made comtrj ...')
        #if os.path.isfile(com_trj) and not os.path.isfile(unwrapped_com_trj)    :
        os.system('gmx trjconv -f {0} -o {1} -pbc nojump'.format(
                com_trj, unwrapped_com_trj))


@Project.operation
@Project.pre.isfile(unwrapped_file)
@Project.post.isfile(msd_file)
def run_msd(job):
    print('Loading trj {}'.format(job))
    top_file = os.path.join(job.workspace(), 'com.gro')
    trj_file = os.path.join(job.workspace(),
            'sample_com_unwrapped.xtc')
    trj = md.load(trj_file, top=top_file)
    #selections = {'all' : trj.top.select('all'),
    #              'ion' : trj.top.select('resname li tf2n'),
    #              'cation': trj.top.select("resname li"),
    #              'anion': trj.top.select("resname tf2n"),
    #              'chloroform': trj.top.select('resname chlor'),
    #              'ch3cn': trj.top.select('resname ch3cn')
    #              }
    selections = {
                  'cation': trj.top.select("resname li"),
                  'anion': trj.top.select("resname tf2n"),
                  'chloroform': trj.top.select('resname chlor'),
                  'ch3cn': trj.top.select('resname ch3cn')
                  }

    for mol, indices in selections.items():
        print('\tConsidering {}'.format(mol))
        if indices.size == 0:
            print('{} does not exist in this statepoint'.format(mol))
            continue
        print(mol)

        sliced = trj.atom_slice(indices)
        D_bar, D_std = _run_multiple(sliced, mol)
        job.document['D_' + mol + '_bar_5'] = D_bar
        job.document['D_' + mol + '_std_5'] = D_std


@Project.operation
@Project.pre.isfile(msd_file)
def run_rdf(job):
    print('Loading trj {}'.format(job))
    if os.path.exists(os.path.join(job.workspace(), 'com.gro')):
        top_file = os.path.join(job.workspace(), 'com.gro')
        trj_file = os.path.join(job.workspace(), 'sample_com.xtc')
        trj = md.load(trj_file, top=top_file, stride=10)

        selections = dict()
        selections['cation'] = trj.topology.select('name li')
        selections['anion'] = trj.topology.select('resname {}'.format(job.statepoint()['anion']))
        selections['acn'] = trj.topology.select('resname ch3cn')
        selections['chlor'] = trj.topology.select('resname chlor')
        selections['all'] = trj.topology.select('all')

        combos = [('cation', 'anion'),
                  ('cation','cation'),
                  ('anion','anion'),
                  ('acn','anion'),
                  ('acn','cation'),
                  ('chlor','anion'),
                  ('chlor','cation'),
                  ('acn', 'chlor')]
        for combo in combos:
            fig, ax = plt.subplots()
            print('running rdf between {0} ({1}) and\t{2} ({3})\t...'.format(combo[0],
                                                                             len(selections[combo[0]]),
                                                                             combo[1],
                                                                             len(selections[combo[1]])))
            r, g_r = md.compute_rdf(trj, pairs=trj.topology.select_pairs(selections[combo[0]], selections[combo[1]]), r_range=((0.0, 2.0)))

            data = np.vstack([r, g_r])
            np.savetxt('txt_files/{}-{}-{}-rdf-{}-{}.txt'.format(job.sp.chlor_conc,
                    job.sp.acn_conc, job.sp.il_conc, combo[0], combo[1]),
                np.transpose(np.vstack([r, g_r])),
                header='# r (nm)\tg(r)')
            ax.plot(r, g_r)
            plt.xlabel('r (nm)')
            plt.ylabel('G(r)')
            plt.savefig(os.path.join(job.workspace(),
                   f'rdf-{combo[0]}-{combo[1]}.pdf'))
            print(' ... done\n')


@Project.operation
@Project.pre.isfile(msd_file)
def run_cond(job):
    if 'D_cation_bar_2' in job.document().keys():
        top_file = os.path.join(job.workspace(), 'sample.gro')
        trj_file = os.path.join(job.workspace(),
                'sample_unwrapped.xtc')
        trj = md.load(trj_file, top=top_file)
        cation = trj.topology.select('name Li')
        cation_msd = job.document()['D_cation_bar_5']
        anion_msd = job.document()['D_anion_bar_5']
        cation_std = job.document()['D_cation_std_5']
        anion_std = job.document()['D_anion_std_5']
        volume = float(np.mean(trj.unitcell_volumes)) * u.nm**3
        volume = volume.to(u.m**3)
        N = len(cation)
        T = job.sp['T']

        conductivity = calc_conductivity(N, volume, cation_msd, anion_msd, T=T)
        std_conductivity = calc_conductivity(N, volume, cation_std, anion_std, T=T)
        job.document['ne_bar_5'] = float(conductivity.value)
        job.document['ne_std_5'] = float(std_conductivity.value)
        #print(std_conductivity)
        print('Conductivity calculated')

@Project.operation
#@Project.pre.isfile(msd_file)
def run_eh_cond(job):
    print(job.get_id())
    top_file = os.path.join(job.workspace(), 'com.gro')
    trj_file = os.path.join(job.workspace(), 'sample_com_unwrapped.xtc')
    trj_frame = md.load_frame(trj_file, top=top_file, index=0)

    trj_ion = trj_frame.atom_slice(trj_frame.top.select('resname li {}'.format(
        job.statepoint()['anion'])))
    charges = get_charges(trj_ion, job.statepoint()['anion'])
    new_charges = list()
    for charge in charges:
        if charge != 1:
            if charge > 0:
                charge = 1
            else:
                charge = -1
            new_charges.append(charge)

    chunk = 200 
    overall_avg = list()
    trj = md.load(trj_file, top=top_file)
    for outer_chunk in range(0, 10000, 2000):
        running_avg = np.zeros(chunk)
        trj_outer_chunk = trj[outer_chunk:outer_chunk+2000]
        for i, start_frame in enumerate(np.linspace(0, 2000, num=500, dtype=np.int)):
            end_frame = start_frame + chunk
            if end_frame < 2000: 
                trj_chunk = trj_outer_chunk[start_frame:end_frame]
                print('\t\t\t...frame {} to {}'.format(start_frame, end_frame))
                if i == 0:
                    trj_time = trj_chunk.time
                trj_slice = trj_chunk.atom_slice(trj_chunk.top.select('resname li {}'.format(
                      job.statepoint()['anion'])))
                M = dipole_moments_md(trj_slice, new_charges)
                intermediate = [np.linalg.norm((M[i] - M[0]))**2 for i in range(len(M))]  
                plt.plot(trj_time-trj_time[0].reshape(-1), intermediate)
                running_avg += intermediate

        y = running_avg / i
        overall_avg.append(y) 
   
    x = (trj_time - trj_time[0]).reshape(-1)
    
    eh_list = list()
    for y in overall_avg: 
        slope, intercept, r_value, p_value, std_error = stats.linregress(
                x[25:], y[25:])
        
        kB = 1.38e-23 * joule / kelvin
        V = np.mean(trj_frame.unitcell_volumes, axis=0) * nanometer ** 3
        T = job.statepoint()['T'] * kelvin
        
        sigma = slope * (elementary_charge * nanometer) ** 2 / picosecond / (6 * V * kB * T)
        seimens = seconds ** 3 * ampere ** 2 / (kilogram * meter ** 2)
        sigma = sigma.in_units_of(seimens / meter)
        eh_list.append(sigma/sigma.unit)

    plt.plot(x[25:], y[25:], color='k', linestyle='solid', label='actual')
    plt.plot(x[25:], (slope*x[25:]) + intercept, color='k', linestyle='--', label='fit')
    plt.savefig('test_2.pdf')

    eh_bar = np.mean(eh_list)
    eh_std = np.std(eh_list)
    print(eh_bar)
    print(eh_std)

    job.document['eh_bar'] = eh_bar
    job.document['eh_std'] = eh_std

@Project.operation
def run_rho(job):
    print('Loading trj {}'.format(job))
    top_file = os.path.join(job.workspace(), 'sample.gro')
    trj_file = os.path.join(job.workspace(), 'sample.xtc')
    trj = md.load(trj_file, top=top_file)

    # Compute density in kg * m ^ -3
    rho = calc_density(trj)

    job.document['rho'] = float(np.mean(rho))

    # Compute and store volume in nm ^ -3
    job.document['volume'] = float(np.mean(trj.unitcell_volumes))


@Project.operation
@Project.pre.isfile(rdf_file)
#@Project.post.isfile(cn_file)
def run_cn(job):
    trj = md.load_frame(os.path.join(job.workspace(), 'sample.trr'),
                  1000,
                  top=os.path.join(job.workspace(), 'sample.gro'))
    volume = trj.unitcell_volumes
    combinations = [['chlor','cation'],
                    ['chlor', 'anion'],
                    ['acn', 'cation'],
                    ['cation', 'anion'],
                    ['acn', 'anion'],
                    ['acn', 'chlor']]
                    
    for combo in combinations:
        if job.sp.il_conc in [0.3, 0.6]:
            continue
        r, g_r = np.loadtxt(os.path.join(job.workspace(),
              'rdf-{}-{}.txt'.format(
               combo[0],combo[1]))).T

        if combo == ['chlor', 'cation']:
            if job.sp.chlor_conc == 0:
                continue
            #chunk = np.where((r>0.6) & (r<0.75))
            chunk = np.where((r>1.0) & (r<1.20))
        elif combo == ['chlor', 'anion']:
            if job.sp.chlor_conc == 0:
                continue
            chunk = np.where((r>0.8) & (r<0.95))
        elif combo == ['acn', 'cation']:
            chunk = np.where((r>0.3) & (r<0.45))
        elif combo == ['acn', 'anion']:
            chunk = np.where((r>0.7) & (r<0.9))
        elif combo == ['cation', 'anion']:
            chunk = np.where((r>0.5) & (r<0.65))
        elif combo == ['acn', 'chlor']:
            if job.sp.chlor_conc == 0:
                continue
            chunk = np.where((r>0.55) & (r<0.75))

        g_r_chunk = g_r[chunk]
        r_chunk = r[chunk]
        if combo == ['chlor', 'cation']:
             n_mols = (job.sp.il_conc * job.sp.n_acn) / job.sp.acn_conc
             #n_mols = (job.sp.chlor_conc * job.sp.n_acn) / job.sp.acn_conc
             rho = n_mols / volume
        elif combo == ['chlor', 'anion']:
             n_mols = (job.sp.il_conc * job.sp.n_acn) / job.sp.acn_conc
             #n_mols = (job.sp.chlor_conc * job.sp.n_acn) / job.sp.acn_conc
             rho = n_mols / volume
        elif combo == ['cation', 'anion']:
             n_mols = (job.sp.il_conc * job.sp.n_acn) / job.sp.acn_conc
             #n_mols = (job.sp.chlor_conc * job.sp.n_acn) / job.sp.acn_conc
             rho = n_mols / volume
        elif combo == ['acn', 'cation']:
             n_mols = (job.sp.il_conc * job.sp.n_acn) / job.sp.acn_conc
             #n_mols = (job.sp.n_acn)
             rho = n_mols / volume
        elif combo == ['acn', 'anion']:
             n_mols = (job.sp.il_conc * job.sp.n_acn) / job.sp.acn_conc
             #n_mols = (job.sp.n_acn)
             rho = n_mols / volume

        N = [np.trapz(4 * rho * np.pi * g_r[:i] * r[:i] **2, r[:i], r) for i in range(len(r))]

        # Store CN near r = 0.8
        index = np.where(g_r == np.amin(g_r_chunk))
        print(f'{job.sp.chlor_conc}-{job.sp.acn_conc}-{job.sp.il_conc}')
        print('combo is {}'.format(combo))
        print('g_r is {}'.format(g_r[index]))
        print('r is {}'.format(r[index]))
        print(N[int(index[0])])
        #if combo == ['acn', 'anion']:
        #     print(job.document()['cn_acn_anion'])
        job.document['cn_{}_{}_2'.format(combo[0], combo[1])] = N[int(index[0])]
        job.document['r_cn_{}_{}_2'.format(combo[0], combo[1])] = float(r[index])
        #fig, ax = plt.subplots()
        #ax.plot(r, g_r)
        #ax.plot(r, N, '--')
        #ax.scatter(r[index], g_r[index], marker='o', color='black')
        #plt.xlabel('r (nm)')
        #plt.ylabel('g(r)')
        #plt.ylim((0,25))
        #plt.savefig(os.path.join(job.workspace(), f'{combo[0]}_{combo[1]}_with_cn.pdf'))

        #
        # Save entire CN
        np.savetxt('txt_files/{}-{}-{}-cn-{}-{}_2.txt'.format(job.sp.chlor_conc,
                        job.sp.acn_conc, job.sp.il_conc, combo[0],combo[1]),
                  np.transpose(np.vstack([r, N])),
                  header='# r (nm)\tCN(r)\tr_index={}\tcn={}'.format(r[index], N[int(index[0])]))
    

def _gromacs_str(op_name, gro_name, sys_name, job):
    """Helper function, returns grompp command string for operation """
    if op_name == 'em':
        mdp = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name))
        cmd = ('gmx grompp -f {mdp} -c gaff.gro -p gaff.top -o {op}.tpr --maxwarn 1 && gmx mdrun -deffnm {op} -ntmpi 1')
    else:
        mdp = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name, job.sp.T))
        cmd = ('gmx grompp -f {mdp} -c {gro}.gro -p gaff.top -o {op}.tpr --maxwarn 1 && gmx mdrun -deffnm {op} -ntmpi 1')
    return workspace_command(cmd.format(mdp=mdp,op=op_name, gro=gro_name, sys=sys_name))

def get_charges(trj, anion):
    charges = np.zeros(shape=(trj.n_atoms))

    for i, atom in enumerate(trj.top.atoms):
        if anion == 'fsi':
            if atom.name == 'fsi':
                charges[i] = -0.6
            elif atom.name == 'li':
                charges[i] = 0.6
        else:
            if atom.name == 'tf2n':
                charges[i] = -0.8
            elif atom.name == 'li':
                charges[i] = 0.8
    return charges

def dipole_moments_md(traj, charges):
    local_indices = np.array([(a.index, a.residue.atom(0).index) for a in traj.top.atoms], dtype='int32')
    local_displacements = md.compute_displacements(traj, local_indices, periodic=False)

    molecule_indices = np.array([(a.residue.atom(0).index, 0) for a in traj.top.atoms], dtype='int32')
    molecule_displacements = md.compute_displacements(traj, molecule_indices, periodic=False)

    xyz = local_displacements + molecule_displacements

    moments = xyz.transpose(0, 2, 1).dot(charges)

    return moments


if __name__ == '__main__':
    Project().main()
