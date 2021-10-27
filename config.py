
import numpy as np

attrib_dict = {'gender': ['male','female'], 'age': ['30-39', '50-59', '18-24', '65+', '25-29', '40-49', '60-64'], 'ethnicity': ['other', 'asian', 'black', 'white', 'latino'], 'region': ['desert_view_highlands', 'lake_los_angeles', 'quartz_hill', 'littlerock', 'acton', 'palmdale', 'northwest_antelope_valley', 'lancaster', 'leona_valley', 'sun_village', 'northwest_palmdale', 'northeast_antelope_valley', 'southeast_antelope_valley']}
rand_seed = 42

#######attrib_dict = {'region': ['desert_view_highlands', 'lake_los_angeles', 'quartz_hill', 'littlerock', 'acton', 'palmdale', 'northwest_antelope_valley', 'lancaster', 'leona_valley', 'sun_village', 'northwest_palmdale', 'northeast_antelope_valley', 'southeast_antelope_valley']}

#######simulations = 10000
simulations = 10000
graph_path = '/Users/sasikanth/Desktop/M.Tect IIT Jodhpur/Sem4/SNA/Project/Code/networks/graph_spa_500_1.pickle'

edge_pp_lb = 0.001
edge_pp_ub = 0.1

rand_diff_prob15 = np.random.uniform(edge_pp_lb, edge_pp_ub)

k = 15
theta = 0.01

nodes=100
attrib_dict_g = {'Gender': ['male', 'female'], 'Region': ['India', 'US'], 'Age': ['20-25', '50-59']}
#attrib_dict_g = {'Region': ['India', 'US']}
#attrib_dict_g = {'Gender': ['male', 'female']}
#attrib_dict_g = {'Age': ['20-25', '50-59']}



