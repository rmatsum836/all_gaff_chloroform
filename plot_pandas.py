import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import signac
from collections import OrderedDict

data = pd.read_csv('project.csv')
data = data[(data.il_conc != 0.3) & (data.il_conc != 0.6)]

def plot_ion_d(data):
    fig, ax = plt.subplots()
    charge_type = 'all_resp'
    for k,v in {'overall_2': 'green', 'bar_5': 'blue'}.items():
        cat_d = data[f'D_cation_{k}'][data.charge_type == charge_type]
        an_d = data[f'D_anion_{k}'][data.charge_type == charge_type]
        acn_conc = data['acn_conc'][data.charge_type == charge_type]
        chlor_conc = data['chlor_conc'][data.charge_type == charge_type]
        ratio = chlor_conc / acn_conc

        plt.scatter(ratio, cat_d, label=f'Li Diffusivity: {charge_type}-{k}', marker='x', color=v)
        plt.scatter(ratio, an_d, label=f'TFSI Diffusivity: {charge_type}-{k}', marker='o', color=v)

    plt.xlabel('chloroform-acetonitrile ratio')
    plt.ylabel('Diffusivity')
    plt.yscale('log')
    plt.ylim([1e-11, 9e-9])
    plt.legend()
    plt.savefig('ion_diffusivity.pdf')

def plot_solvent_d(data):
    fig, ax = plt.subplots()
    #for charge_type in ['all_resp', 'resp']:
    charge_type = 'all_resp'
    for k,v in {'overall_2': 'green', 'bar_5': 'blue'}.items():
        acn_d = data[f'D_ch3cn_{k}'][data.charge_type == charge_type]
        chlor_d = data[f'D_chloroform_{k}'][data.charge_type == charge_type]
        acn_conc = data['acn_conc'][data.charge_type == charge_type]
        chlor_conc = data['chlor_conc'][data.charge_type == charge_type] 
        ratio = chlor_conc / acn_conc

        plt.scatter(ratio, acn_d, label=f'ACN Diffusivity: {charge_type}', marker='x', color=v)
        plt.scatter(ratio, chlor_d, label=f'CHLOR Diffusivity: {charge_type}', marker='o', color=v)
    plt.xlabel('chloroform-acetonitrile ratio')
    plt.ylabel('Diffusivity')
    plt.yscale('log')
    plt.ylim([1e-11, 9e-9])
    plt.legend()
    plt.savefig('solvent_diffusivity.pdf')

def plot_ionicity(data):
    fig, ax = plt.subplots()
    for charge_type in ['all_resp']:
        eh = data['eh_bar'][data.charge_type == charge_type]
        #ne = data['ne_bar'][data.charge_type == charge_type]
        ne = data['ne_bar_5'][data.charge_type == charge_type]
        acn_conc = data['acn_conc'][data.charge_type == charge_type]
        chlor_conc = data['chlor_conc'][data.charge_type == charge_type] 
        ratio = chlor_conc / acn_conc

        plt.scatter(ratio, eh/ne, label=f'Ionicity: {charge_type}', marker='x')
        plt.xlabel('chloroform-acetonitrile ratio')
        plt.ylabel('Ionicity')
        plt.legend()
        plt.savefig('ionicity.pdf')

def plot_conductivity(data):
    fig, ax = plt.subplots()
    charge_type = 'all_resp'
    for k,v in {'overall': 'green', 'bar_5': 'blue'}.items():
        eh = data['eh_bar'][data.charge_type == charge_type]
        eh_std = data['eh_std'][data.charge_type == charge_type]
        ne = data[f'ne_{k}'][data.charge_type == charge_type]
        #ne_std = data['ne_std_5'][data.charge_type == charge_type]
        acn_conc = data['acn_conc'][data.charge_type == charge_type]
        chlor_conc = data['chlor_conc'][data.charge_type == charge_type] 
        ratio = chlor_conc / acn_conc

        #plt.errorbar(ratio, eh, yerr=eh_std, label=f'EH conductivity: {charge_type}', marker='o', ls='none')
        #plt.errorbar(ratio, ne, yerr=ne_std, label=f'NE conductivity: {charge_type}', marker='x', ls='none')
        plt.plot(ratio, ne, ls='none', marker='o')

    
        plt.xlabel('chloroform-acetonitrile ratio')
        plt.ylabel('Conductivity')
        plt.legend()
        plt.savefig('conductivity.pdf')

def plot_anion_cn(data):
    fig, ax = plt.subplots()
    acn_an = data['cn_acn_anion_2']
    chlor_an = data['cn_chlor_anion_2']
    ion_an = data['cn_cation_anion_2']
    acn_chlor = data['cn_acn_chlor_2']
    acn_conc = data['acn_conc']
    chlor_conc = data['chlor_conc']
    ratio = chlor_conc / acn_conc

    plt.scatter(ratio, acn_an, label='ACN-TFSI', marker='o')
    plt.scatter(ratio, chlor_an, label='Chloroform-TFSI', marker='x')
    plt.scatter(ratio, ion_an, label='Li-TFSI', marker='v')
    plt.scatter(ratio, acn_chlor, label='ACN-Chloroform', marker='>')

    plt.xlabel('chloroform-acetonitrile ratio')
    plt.ylabel('Coordination Number')
    plt.legend()
    plt.savefig('anion_cn_2.pdf')

def plot_cation_cn(data):
    fig, ax = plt.subplots()
    acn_an = data['cn_acn_cation_2']
    chlor_an = data['cn_chlor_cation_2']
    ion_an = data['cn_cation_anion_2']
    acn_chlor = data['cn_acn_chlor_2']
    acn_conc = data['acn_conc']
    chlor_conc = data['chlor_conc']
    ratio = chlor_conc / acn_conc

    plt.scatter(ratio, acn_an, label='ACN-Li', marker='o')
    plt.scatter(ratio, chlor_an, label='Chloroform-Li', marker='x')
    plt.scatter(ratio, ion_an, label='Li-TFSI', marker='v')
    plt.scatter(ratio, acn_chlor, label='ACN-Chloroform', marker='>')

    plt.xlabel('chloroform-acetonitrile ratio')
    plt.ylabel('Coordination Number')
    plt.legend()
    plt.savefig('cation_cn_2.pdf')

def plot_r_distances(data):
    fig, ax = plt.subplots()
    combinations = [['chlor','cation'],
                    ['chlor', 'anion'],
                    ['acn', 'cation'],
                    ['cation', 'anion'],
                    ['acn', 'anion'],
                    ['acn', 'chlor']]

    acn_conc = data['acn_conc']
    chlor_conc = data['chlor_conc']
    ratio = chlor_conc / acn_conc

    for combo in combinations:
        combo_data = data['r_cn_{}_{}_2'.format(combo[0], combo[1])]

        plt.scatter(ratio, combo_data, label='{}-{}'.format(combo[0], combo[1]))

    plt.xlabel('chloroform:acetonitrile')
    plt.ylabel('Distance used for Coordination Numbers (nm)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig('cn_distances_2.pdf')

def plot_ratios(data):
    fig, ax = plt.subplots()
    charge_type = 'all_resp'
    #cat_d = data['D_cation_bar_5'][data.charge_type == charge_type]
    #an_d = data['D_anion_bar_5'][data.charge_type == charge_type]
    #acn_d = data['D_ch3cn_bar_5'][data.charge_type == charge_type]
    #chlor_d = data['D_chloroform_bar_5'][data.charge_type == charge_type]
    cat_d = data['D_cation_overall_2'][data.charge_type == charge_type]
    an_d = data['D_anion_overall_2'][data.charge_type == charge_type]
    acn_d = data['D_ch3cn_overall_2'][data.charge_type == charge_type]
    chlor_d = data['D_chloroform_overall_2'][data.charge_type == charge_type]
    acn_conc = data['acn_conc'][data.charge_type == charge_type]
    chlor_conc = data['chlor_conc'][data.charge_type == charge_type]
    ratio = chlor_conc / acn_conc

    #jplt.scatter(ratio, an_d/cat_d, label='tfsi/li')
    #plt.scatter(ratio, chlor_d/acn_d, label='chlor/acn')
    plt.scatter(ratio, chlor_d/cat_d, label='chlor/li')
    plt.scatter(ratio, acn_d/cat_d, label='acn/li')
    plt.scatter(ratio, acn_d/an_d, label='acn/tfsi')
    #plt.scatter(ratio, an_d, label=f'TFSI Diffusivity: {charge_type}-{k}', marker='o', color=v)

    plt.xlabel('chloroform-acetonitrile ratio')
    plt.ylabel('Diffusivity')
    #plt.yscale('log')
    plt.ylim([2, 10])
    plt.legend()
    plt.savefig('acn_diffusivity_ratios.pdf')

plot_ion_d(data)
plot_solvent_d(data)
plot_ionicity(data)
plot_conductivity(data)
plot_cation_cn(data)
plot_anion_cn(data)
plot_r_distances(data)
plot_ratios(data)
