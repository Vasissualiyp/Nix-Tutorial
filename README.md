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

Once you're done experimenting, exit the shell and navigate back with 
```
cd ..
```

# 3. Python Nix shell

Making cow say stuff is amusing, but quite useless. How about we do something productive?

Navigate to the directory of the 3rd activity and look at contents of `shell.nix`.
(If you don't know how, try running `nano shell.nix`. If you don't have nano,
running this shell should give it to you).

In this shell, by default, we have python 3.12 with numpy. I have a few other packages
that we usually use commented out - you can comment them back by removing `#` symbol.

This shell also includes an example of how you can define variables with Nix.
You do so in `let` block, and the actual code that you want Nix to run you put 
in `in` block. For instance, this shell only pulls `nano` and our python 
environment, that we define in `pythonEnv` variable.

Try running `python test.py`. This test script will tell you version of your python
and some packages that we have/don't have in the current shell.
You can also try running this test script outside of nix shell to see which of these 
packages you have installed on your system, and you can compare to the packages
accessible from the nix shell.

## Task 3.1

Try changing the shell to use python 3.10 instead of 3.12

## Task 3.2

Try having both python 3.10 and 3.12 accessible from a single shell
(Hint: you can use different python binaries via `python3.10` or `python3.12` 
instead of `python`)

## Task 3.3

Try adding `mpi` as another (non-python) package. It will automatically install MPI 
into your shell, taking care of library paths, environment variables, etc.!

You can check that it is working by running:
```
mpirun -np 2 python test.py
```

# 4. Nix flakes

Those who pay very close attention might have noticed that at no point we declare
our versions of packages when working with shells, thus Nix's claim of reproducibility
doesn't hold: if you were to run the same shell a year later, it will give
you the newest package versions... But all is not lost!

Nix flakes are an "experimental" feature (but everyone mostly uses them and not 
the shells), that "locks" the versions of packages, allowing full bit-by-bit 
reproducibility that Nix promotes. To check out an example of how a flake for the 
same shell from previos activity would look like, navigate to the 4th activity
directory, and run:
```
nix develop --experimental-features 'nix-command flakes'
```
(If you enable flakes in your nix config, you can get away with just `nix develop`,
but we won't go into that today).

You then should be able to run the `test.py` script the same way.

Inspecting the `flake.nix`, you can see a lot more boilerplate...
Which is needed for reproducibility and convenience.
You can also inspect the `flake.lock` file - it gets created when you first 
develop the flake, and it "locks" the versions of packages.
So if you were to develop this flake 10 years later, you will get
bit-by-bit exactly the same environment (assuming that you can still
download all the packages from their mirrors, which is the case in 99.99999% 
of scenarios, unless our civilization collapses)

I also added a message that you get when you just start the flake with `shellHook`.
There you can personalize it, or add environment variables and other things you 
might need in your environment.
