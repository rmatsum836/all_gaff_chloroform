import pandas as pd
import numpy as np
import signac

project = signac.get_project()
df = pd.DataFrame(project.index())
df = df.set_index(['_id'])
statepoints = {doc['_id']: doc['statepoint'] for doc in project.index()}
df_index = pd.DataFrame(statepoints).T.join(df)
df_index.to_csv('project.csv')
