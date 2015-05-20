{ system ? builtins.currentSystem
, networkConfig
, checkConfigurationOptions ? true
, networkState ? null
, fn ? null
}:

with import <nixpkgs/nixos/lib/testing.nix> { inherit system; };
with pkgs;
with lib;
let
  maybe = x: expr: if x == null then null else expr;
  network = callPackage networkConfig {};
  getMachineArgs = machineName: (if networkState == null then { machines = null; self = null;} else
    let state = import networkState;
    in { self = (getAttr machineName state.machines); } // state
  );

  # like callPackage, but only includes specific arguments
  callFunction = f: args: f (builtins.intersectAttrs (builtins.functionArgs f) args);
  maybeCall = f: a: if builtins.isFunction f then f a else f;

  defaultConf = (network.defaults or {});
  machines = mapAttrs (machineName: machineConf: (
    let machineArgs = getMachineArgs machineName; in
    import <nixpkgs/nixos/lib/eval-config.nix> {
      modules = [
        ({imports = [ machineConf ]; key = "${machineName}"; _file = networkConfig;})
        ({imports = [ defaultConf ]; key = "defaults:${machineName}"; _file = networkConfig;})

        { key = "nixops-stuff";
          # Make NixOps's deployment.* options available.
          imports = [ ./options.nix ];
          deployment.state = machineArgs;
          # Provide a default hostname and deployment target equal
          # to the attribute name of the machine in the model.
          # networking.hostName = mkOverride 900 machineName;
          # deployment.targetHost = mkOverride 900 machineName;
          # deployment.sshPort = mkOverride 900 22; # XXX get form services.sshd?
          # environment.checkConfigurationOptions = mkOverride 900 checkConfigurationOptions;
          services.openssh.enable = true;
        }
      ];
      extraArgs = machineArgs;
    }
  ).config) network.machines;

in {
  inherit networkConfig;
  inherit machines;

  # Phase 1: evaluate only the deployment attributes
  info = {
    provider = network.provider;
    machines = mapAttrs (machineName: machine: (machine.deployment or {}) // {hooks= null;}) machines;
    bootstrap = network.bootstrap or null;
  };

  # phase 1.5: export bootstrap images for use on various providers
  bootstrap = {
    libvirtd = {size}: import ./libvirtd-image.nix {
      inherit size system;
      inherit (network.provider) authorizedKeys;
    };
    gce = {
      image = {diskSize}: (import <nixpkgs/nixos/lib/eval-config.nix> {
        modules = [
          (import ./google-compute-image.nix {
            diskSize = "${builtins.toString diskSize}G";
            overrides = network.bootstrap.config;
          })
        ];

      }).config.system.build.googleComputeImage;
    };
  };

  # Phase 2: after infrastructure is defined, we can compute both hooks and
  # full system config.

  hooks = mapAttrs (machineName: machineConf: (
    # NOTE: there are nested sources of sortOrder:
    # hooks.sortOrder
    # hooks.<action>.sortOrder OR hooks.custom.<action>.sortOrder
    let
      hooks = machineConf.deployment.hooks;
      baseOrder = hooks.order or null;
      applyOrdering = hookName: hook:
        let order = maybeCall (hook.order or baseOrder) {
          hookName = hookName;
          hook = hook;
          machineName = machineName;
          machine = machineConf;
        }; in
        hook // {
          order = {
            parallel = order.parallel or 0;
            sort = order.sort or null;
            batch = order.batch or null;
          };
        };
    in
    mapAttrs (hookName: hook:
      if hookName == "custom" then (
        # recurse one level deeper
        mapAttrs applyOrdering hook
      ) else (
        applyOrdering hookName hook
      )
    # filter out top-level hook attributes which aren't hooks:
    ) (filterAttrs (k: v: k != "order") hooks)
  )) machines;

  customHookNames = mapAttrs (machineName: machineConf: (
    attrNames (machineConf.deployment.hooks.custom or {})
  )) machines;

  system = mapAttrs (machineName: machineConf: (
    machineConf.system.build.toplevel
  )) machines;

  config = runCommand "mapnix-machines"
    { preferLocalBuild = true; }
    ''
      mkdir -p $out
      ${concatStringsSep "\n" (attrValues (mapAttrs (n: v: ''
        ln -s ${v.system.build.toplevel} $out/${n}
      '') machines))}
    '';

  evalFn = callFunction fn {inherit machines pkgs lib;};
}
