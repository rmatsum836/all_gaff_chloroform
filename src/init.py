#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import argparse

import signac


def main(args):
    project = signac.init_project('all_gaff_chloroform')
    statepoints_init = []
    temperatures = [298]
    anions = ['tf2n']
    charge_type = ['all_resp']

    # Concentrations are ordered as : [ACN: CHLOR: SIS]
    concentrations = [
                      [1, 0, 0.33],
                      [1, 3, 0.33],
                      [1, 1, 0.33],
                      [2, 3, 0.66],
                      [3, 1, 1],
                      [3, 2, 1],
                     ]
                   
    for anion in anions:
        for charge in charge_type:
            for conc in concentrations:
                if conc[1] == 10:
                    n_acn = 500
                elif conc == [3, 1, 1]:
                    n_acn = 1750
                elif conc == [3, 2, 1]:
                    n_acn = 1500
                elif conc == [1, 1, 0.33]:
                    n_acn = 1500
                elif conc == [1, 3, 0.33]:
                    n_acn = 1000
                elif conc == [1, 0, 0.33]:
                    n_acn = 2000
                elif conc == [2, 3, 0.66]:
                    n_acn = 1200
                for temp in temperatures:
                    statepoint = dict(
                                anion=anion,
                                T=temp,
                                acn_conc=conc[0],
                                chlor_conc=conc[1],
                                il_conc=conc[2],
                                n_acn=n_acn,
                                charge_type=charge
                                )
                    project.open_job(statepoint).init()
                    statepoints_init.append(statepoint)

    # Writing statepoints to hash table as a backup
    project.write_statepoints(statepoints_init)


if __name__ == '__main__':
     parser = argparse.ArgumentParser(
         description="Initialize the data space.")
     parser.add_argument(
         '-n', '--num-replicas',
         type=int,
         default=1,
         help="Initialize multiple replications.")
     args = parser.parse_args()
  
     logging.basicConfig(level=logging.INFO)
     main(args)
