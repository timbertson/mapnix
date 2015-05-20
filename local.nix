{pkgs ? import <nixpkgs> {}}:
with pkgs;
lib.overrideDerivation (callPackage ./default.nix {}) (base: {
	src = ./bin/..;
})
