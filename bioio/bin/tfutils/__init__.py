# %%
import click

from . import count, filter_regex

# %%
@click.group()
def main():
    pass

# %%
main.add_command(count.main, name='count')
main.add_command(filter_regex.main, name='filter-regex')

# %%
if __name__ == '__main__':
    main()
