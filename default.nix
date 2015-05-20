{pkgs ? import <nixpkgs> {},
	lib ? pkgs.lib,
	stdenv ? pkgs.stdenv,
	fetchurl ? pkgs.fetchurl,
	fetchgit ? pkgs.fetchgit,
	python ? pkgs.python,
	pythonPackages ? pkgs.pythonPackages,
	makeWrapper ? pkgs.makeWrapper
}:
stdenv.mkDerivation {
	name = "mapnix-env";
	buildInputs = [ makeWrapper pythonPackages.wrapPython ];
	src = fetchgit {
		url = "https://github.com/gfxmonk/mapnix.git";
		rev = "07ea26454a15b3076e6f099c55a0a8b51d3c4b32";
		sha256="b9833bc41554b4cb5e5046dc830e0f59a7e2f3b8c477df59d20bef9cb5c56abd";
	};
	pythonPath = [
		pythonPackages.pycrypto

		# packaged libcloud is only 0.14:
		# pythonPackages.libcloud
		(lib.overrideDerivation pythonPackages.libcloud (base: {
			name = "libcloud-0.16.0";
			src = fetchurl {
				url = mirror://apache/libcloud/apache-libcloud-0.16.0.tar.bz2;
				sha256 = "0yhdkjsxxhq5lywpvy3bm60s48b3lqppamrqwm390ca0qqkx0nzn";
			};
		}))
	];
	installPhase = ''
		libdest=$out/lib/${python.libPrefix}/site-packages
		mkdir $out
		mkdir -p $libdest

		cp -Pr bin $out/bin
		cp -Pr mapnix $libdest/
	'';
	fixupPhase = "wrapPythonPrograms";
}
