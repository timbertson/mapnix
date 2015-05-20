{pkgs ? import <nixpkgs> {}}:
with pkgs;
lib.overrideDerivation (callPackage ./default.nix {}) (base: {
	PYTHONPATH = pkgs.lib.concatStringsSep ":" (
		(map
			(pth: "${pth}/lib/${python.libPrefix}/site-packages")
			base.pythonPath
		) ++ [(builtins.getEnv "PWD")]
	);
})
