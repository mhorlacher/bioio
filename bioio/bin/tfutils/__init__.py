# %%
import click

from . import count

# %%
@click.group()
def main():
    pass

# %%
main.add_command(count.main, name='count')

# %%
if __name__ == '__main__':
    main()
