import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import signac
from collections import OrderedDict

data = pd.read_csv('project.csv')
data = data[(data.il_conc != 0.3) & (data.il_conc != 0.6)]

def plot_ion_d(data):
    fig, ax = plt.subplots()
    for charge_type in ['all_resp', 'resp']:
        cat_d = data['D_cation_bar_2'][data.charge_type == charge_type]
        an_d = data['D_anion_bar_2'][data.charge_type == charge_type]
        acn_conc = data['acn_conc'][data.charge_type == charge_type]
        chlor_conc = data['chlor_conc'][data.charge_type == charge_type]
        ratio = chlor_conc / acn_conc

        plt.scatter(ratio, cat_d, label=f'Li Diffusivity: {charge_type}', marker='x')
        plt.scatter(ratio, an_d, label=f'TFSI Diffusivity: {charge_type}', marker='o')

    plt.xlabel('chloroform-acetonitrile ratio')
    plt.ylabel('Diffusivity')
    plt.yscale('log')
    plt.ylim([1e-11, 9e-9])
    plt.legend()
    plt.savefig('ion_diffusivity.pdf')

def plot_solvent_d(data):
    fig, ax = plt.subplots()
    for charge_type in ['all_resp', 'resp']:
        acn_d = data['D_ch3cn_bar_2'][data.charge_type == charge_type]
        chlor_d = data['D_chloroform_bar_2'][data.charge_type == charge_type]
        acn_conc = data['acn_conc'][data.charge_type == charge_type]
        chlor_conc = data['chlor_conc'][data.charge_type == charge_type] 
        ratio = chlor_conc / acn_conc

        plt.scatter(ratio, acn_d, label=f'ACN Diffusivity: {charge_type}', marker='x')
        plt.scatter(ratio, chlor_d, label=f'CHLOR Diffusivity: {charge_type}', marker='o')
    plt.xlabel('chloroform-acetonitrile ratio')
    plt.ylabel('Diffusivity')
    plt.yscale('log')
    plt.ylim([1e-11, 9e-9])
    plt.legend()
    plt.savefig('solvent_diffusivity.pdf')

def plot_ionicity(data):
    fig, ax = plt.subplots()
    for charge_type in ['all_resp', 'resp']:
        eh = data['eh_conductivity'][data.charge_type == charge_type]
        ne = data['ne_conductivity'][data.charge_type == charge_type]
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
    for charge_type in ['all_resp', 'resp']:
        eh = data['eh_conductivity'][data.charge_type == charge_type]
        ne = data['ne_conductivity'][data.charge_type == charge_type]
        acn_conc = data['acn_conc'][data.charge_type == charge_type]
        chlor_conc = data['chlor_conc'][data.charge_type == charge_type] 
        ratio = chlor_conc / acn_conc

        plt.scatter(ratio, eh, label=f'EH conductivity: {charge_type}', marker='o')
        plt.scatter(ratio, ne, label=f'NE conductivity: {charge_type}', marker='x')

    
        plt.xlabel('chloroform-acetonitrile ratio')
        plt.ylabel('Conductivity')
        plt.legend()
        plt.savefig('conductivity.pdf')
   
plot_ion_d(data)
plot_solvent_d(data)
plot_ionicity(data)
plot_conductivity(data)
