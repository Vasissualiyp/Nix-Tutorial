{
  description = "Activity 4 flake";

  inputs = {
	nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
	flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
	flake-utils.lib.eachDefaultSystem (system:
	  let
		pkgs = import nixpkgs {
		  inherit system;
		  config = {
			allowUnfree = true;
		  };
		};

		####### Boilerplate ends here #######

		pythonEnv = pkgs.python312Packages.python.withPackages (ps: with ps; [
		  numpy
		]);
	  in
	  {
		devShell = pkgs.mkShell {
		  buildInputs = with pkgs; [
			pythonEnv
			nano
		  ];
		  shellHook = ''
		  export PYVERSION=$(python --version)
		  echo "Welcome to nix flake environment!"
		  echo "Python version is: $PYVERSION"
		  '';
		};
	  }
	);
}
