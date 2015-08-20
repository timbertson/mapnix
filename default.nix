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
	name = "mapnix";
	buildInputs = [ makeWrapper pythonPackages.wrapPython ];
	src = fetchgit {
		url = "https://github.com/gfxmonk/mapnix.git";
		rev = "740a7ad66188040f0bb82871c381029b6b610fb8";
		sha256="9b2be2520543abc80951a73942ad2cc1c135db3676bae7531f74e5197fc45100";
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
