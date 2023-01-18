# %%
import click

# %%
from .bin import serialize

# %%
@click.group()
def main():
    pass

# %%
main.add_command(serialize.main, name='serialize')

# %%
if __name__ == '__main__':
    main()