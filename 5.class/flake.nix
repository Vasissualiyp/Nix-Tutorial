{
  description = "PeakPatch developement environment";

  inputs = {
	nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
	flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
	flake-utils.lib.eachDefaultSystem (system:
	  let
		pkgs = import nixpkgs {
		  inherit system;
		  config = {
			allowUnfree = true;
		  };
		};
		class = pkgs.python312Packages.buildPythonPackage rec {
		  pname = "classy";
		  version = "3.2.3.2";
		  #format = "wheel";
		  src = pkgs.python312Packages.fetchPypi{
			inherit pname;
			inherit version;
			sha256 = "sha256-GxaqQaRABTxhVXDyAYykTVdobRm8rZbZ0wPdfRU23Rc=";
		  };

		  format = "other";

		  nativeBuildInputs = with pkgs; [ python312Packages.numpy 
										   #python312Packages.wheel
										   python312Packages.cython
										   python312Packages.scipy
										   python312Packages.setuptools
										   #python312Packages.setuptools_scm
		  ];
		
		  buildInputs = [
			pkgs.python312Packages.numpy
			pkgs.gsl
		  ];

		  buildPhase = ''
			export CLASS_PKG_PATH="$out/lib/python3.12/site-packages/class_public"
			#export CLASSDIR="$CLASS_PKG_PATH"
			echo "Makefile is located at: $(pwd)"
			${pkgs.python312Packages.python.interpreter} setup.py build
		  '';

		  # Set environment variable to help locate CLASS sources
		  preBuild = ''
			export CLASS_DIR="$PWD"
		  '';

		  patchPhase = ''
			echo "Patching Makefile to set MDIR to installation path."
			substituteInPlace $(pwd)/class_public/Makefile \
			  --replace 'CLASSDIR ?= $(MDIR)' "CLASSDIR := $out/lib/python3.12/site-packages/class_public"
		  '';
		
		  # Override the installPhase to include the external data files
		  installPhase = ''
			  ${pkgs.python312Packages.python.interpreter} setup.py install --prefix=$out --single-version-externally-managed --record=record.txt
		  '';
		  doCheck = false;
		  pythonImportsCheck = [ "classy" ];
		};


		python = pkgs.python312Packages.python;
		pythonEnv = (python.withPackages (ps: with ps; [
		  numpy
		  scipy
		  matplotlib
		  class
		]));
	  in
	  {
		devShell = pkgs.mkShell {
		  buildInputs = with pkgs; [
			pythonEnv
			feh
			nano
		  ];
		  shellHook = ''
		  echo "Welcome to nix shell with class!"
		  '';
	  };
	}
  );
}
