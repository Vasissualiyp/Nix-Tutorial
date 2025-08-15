# August 15th 2025 Cosmolunch: Nix Hands-On Activity

Copy this repository with: 
```
git clone https://github.com/Vasissualiyp/Nix-Tutorial.git
```
Then navigate to it:
```
cd Nix-Tutorial
```
And you should be all set!

You also will need to install Nix package manager onto your OS by following instructions
[nixos.org/download](here).

# 0. Check for correct installation

After you have installed Nix package manager, check that it is working by running:

```
nix run nixpkgs#hello
```

If everything is working, you should get "Hello, world!" in your terminal.

# 1. Nix shell

You can run packages in a nix shell (think of it as a virtual environment, but for all 
software, not just python), by specifying which packages you need:

```
nix-shell -p <package names>
```

For instance:

```
nix-shell -p cowsay
```

Try running `cowsay "Hi"` before activating the shell. It will probably fail, since you 
don't have the package installed. But if you try running it after `nix-shell` command
above, it would work. For fun, try passing a flag `-r` to get a random "cow".

To exit the shell, you can type ~~:q~~ `exit` in your terminal. Try running
`cowsay "Hi"` again. It should fail - the package wasn't installed onto your system,
only into nix-store, and will get removed completely from your machine next time 
you perform garbage collection of nix store.

# 2. Declarative Nix shell

While certainly cool, typing all the packages you need for every project can get
very annoying very fast. Let's try making a declarative shell, which would 
automatically pick the packages you need.

Navigate to the directory of the 2nd activity with:
```
cd 2.declarative_nix_shell
```

Then run `nix-shell`. It should pull `cowsay` package, and now you shold be able to
use it the same way as in the previous activity. 

You can see syntax for nix shell in `shell.nix`. Try adding your own packages!
