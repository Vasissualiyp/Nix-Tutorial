{ pkgs ? import <nixpkgs> { }, }:
let
  pythonEnv = (pkgs.python312Packages.python.withPackages (ps: with ps; [
    numpy
    #matplotlib
    #scipy
    #pandas
    #astropy
  ]));
in
pkgs.mkShell {
  packages = with pkgs; [
    pythonEnv
    nano
  ];
}
