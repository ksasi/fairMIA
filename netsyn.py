
import networkx as nx
import NetworkGenerator as gen
# nx.barabasi_albert_graph

nodes=100
hi = gen.barabasiAlbertAttributeModel(nodes, 4)
hi.add_choice_attribute('Gender', ['male', 'female'], weights=[50, 50],d_index=3)
hi.add_choice_attribute('Region', ['India', 'US'], weights=[90,10], d_index=2)
hi.add_range_attribute('Age',10,20,1,d_index=1)

hi.show_attributes()
G = hi.generate_graph()
print(nx.info(hi.G))
import matplotlib.pyplot as plt

plt.figure(1)
for i in range(nodes):
    plt.scatter(i, hi.G.nodes[i]['Age'])
    # print(hi.G.nodes[i]['Age'])  # if hi.G.nodes[i]['Age'] > 1 else print('',end='')
plt.figure(2)
count = {}
for i in range(nodes):
    plt.scatter(i, hi.G.nodes[i]['Gender'])
    if hi.G.nodes[i]['Gender'] in count:
        count[hi.G.nodes[i]['Gender']] += 1
    else:
        count[hi.G.nodes[i]['Gender']] = 1
plt.figure(3)

for i in range(nodes):
    plt.scatter(i, hi.G.nodes[i]['Region'])
    if hi.G.nodes[i]['Region'] in count:
        count[hi.G.nodes[i]['Region']] += 1
    else:
        count[hi.G.nodes[i]['Region']] = 1
plt.show()
print('lowest', count)

plt.figure()
pos_nodes = nx.spring_layout(G)
nx.draw(G, pos_nodes, with_labels=True)

pos_attrs = {}
for node, coords in pos_nodes.items():
    pos_attrs[node] = (coords[0], coords[1] + 0.08)

node_attrs = nx.get_node_attributes(G, 'Age')
custom_node_attrs = {}
for node, attr in node_attrs.items():
    custom_node_attrs[node] = str(attr)

nx.draw_networkx_labels(G, pos_attrs, labels=custom_node_attrs)
plt.show()
