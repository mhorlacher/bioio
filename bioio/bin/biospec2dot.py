# %%
import click

from bioio.dataspec.loader import load_biospec
from bioio.utils import flatten_dict

# %%
def unique_edges(dataset):
    visited = set()
    edges = []
    def dfs(dataset):
        nonlocal edges
        nonlocal visited

        visited.add(id(dataset))
        
        for parent in dataset._inputs():
            edges.append([f'{parent._name}', f'{dataset._name}'])
            if id(parent) not in visited:
                dfs(parent)
    dfs(dataset)
    return edges

# %%
def dataset_to_dot(data):
    print('digraph test {')
    dot_edges = list()
    flattened_data_dict = flatten_dict(data)
    for key, dataset in flattened_data_dict.items():
        edges = unique_edges(dataset)
        edges.append([edges[0][1], key])
        for edge in edges:
            edge = list(map(lambda x: f'"{x}"', edge))
            dot_edges.append(('\t' + ' -> '.join(edge) + ';'))
    
    for dot_edge in set(dot_edges):
        print(dot_edge)

    print()
    for key in flattened_data_dict.keys():
        print(f'\t"{key}" [color="red"]')

    print('}')

# %%
@click.command()
@click.argument('biospec')
def main(biospec, ):
    data = load_biospec(biospec, to_dataset=False)
    dataset_to_dot(data)


# %%
if __name__ == '__main__':
    main()