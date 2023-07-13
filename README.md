# MSc Thesis Repo

## Structure:
Project is broken down into smaller packages that can be locally installed, to manage dependencies.
 
All Files follow the structure of 

```Python 
import 


class Types

def highest_order_fn

def lower_order_fn

```

## Jax Style: 
use `jax.lax.scan` as much as possible. Makes expression of statefull iteration fast, as long as state is managed with tuples.
only use one dept of scan per function. To make debugging easier.

