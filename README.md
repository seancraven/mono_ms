# MSc Thesis Repo

## Structure:
Project is broken down into smaller packages that can be locally installed, to manage dependencies.
 
All Files follow the structure of 

```Python 
import 


class Types:
    often_named_tuple
    to_pass_state

def highest_order_fn(config):
    return fn

def lower_order_fn(config):
    return fn

```

## Jax Style: 
use `jax.lax.scan` as much as possible. Makes expression of statefull iteration fast, as long as state is managed with tuples.
only use one dept of scan per function. To make debugging easier.

